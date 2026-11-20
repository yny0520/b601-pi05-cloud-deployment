#!/usr/bin/env python3
"""B601-RS (RobStride) scanner and guarded zero calibration tool.

The default command is read-only.  The ``calibrate`` command writes the
RobStride zero reference and stores parameters only after explicit flags and
an interactive confirmation.  It never enables motors or sends a position
target.

Examples::

    python3 b601_rs_calibrate.py scan --channel can0
    python3 b601_rs_calibrate.py snapshot --channel can0 --out rs-before.json
    python3 b601_rs_calibrate.py calibrate --channel can0 --backup rs-before.json \
        --confirm-mechanical-zero --confirm-write
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


HOST_ID = 0xFD
RESPONDER_IDS = {0xFD, 0xFE, 0xFF}
MOTOR_IDS = tuple(range(1, 8))
MOTOR_MODELS = {1: "rs-06", 2: "rs-06", 3: "rs-06", 4: "rs-00", 5: "rs-00", 6: "rs-00", 7: "rs-00"}
JOINT_NAMES = {
    1: "shoulder_pan",
    2: "shoulder_lift",
    3: "elbow_flex",
    4: "wrist_flex",
    5: "wrist_yaw",
    6: "wrist_roll",
    7: "gripper",
}

# Read-only parameters useful for a before/after backup.  The tool never
# writes these values; zeroing uses the dedicated protocol command instead.
PARAMS: tuple[tuple[int, str, str], ...] = (
    (0x7005, "i8", "run_mode"),
    (0x700B, "f32", "limit_torque"),
    (0x7017, "f32", "limit_spd"),
    (0x7018, "f32", "limit_cur"),
    (0x7024, "f32", "vel_max"),
    (0x7029, "u8", "zero_sta"),
    (0x702A, "u8", "damper"),
    (0x702B, "f32", "add_offset"),
)


def die(message: str) -> "NoReturn":
    print(f"错误：{message}", file=sys.stderr)
    raise SystemExit(2)


def import_motorbridge() -> tuple[Any, Any, Any]:
    try:
        from motorbridge import CallError, Controller, MotorBridgeError  # type: ignore
    except ImportError as exc:
        die(
            "找不到 motorbridge。先安装 ARM64 wheel：\n"
            "  python3 -m pip install "
            "/home/jetson/Desktop/b601-rs/motorbridge-assets/"
            "motorbridge-0.5.6-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl"
            f"\n原始错误：{exc}"
        )
    return Controller, CallError, MotorBridgeError


def parse_id(value: str) -> int:
    try:
        result = int(value, 0)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"无效 ID：{value}") from exc
    if not 1 <= result <= 255:
        raise argparse.ArgumentTypeError("电机 ID 必须在 1..255")
    return result


def read_param(motor: Any, param_id: int, kind: str) -> int | float:
    if kind == "i8":
        return motor.robstride_get_param_i8(param_id, 700)
    if kind == "u8":
        return motor.robstride_get_param_u8(param_id, 700)
    return motor.robstride_get_param_f32(param_id, 700)


def state_dict(motor: Any) -> dict[str, Any] | None:
    try:
        motor.request_feedback()
        state = motor.get_state()
    except Exception:
        return None
    if state is None:
        return None
    return {
        "can_id": state.can_id,
        "arbitration_id": state.arbitration_id,
        "status_code": state.status_code,
        "position_rad": state.pos,
        "position_deg": math.degrees(state.pos),
        "velocity_rad_s": state.vel,
        "torque": state.torq,
        "temperature_mos_c": state.t_mos,
        "temperature_rotor_c": state.t_rotor,
    }


def motor_snapshot(motor: Any, motor_id: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "motor_id": motor_id,
        "joint": JOINT_NAMES[motor_id],
        "model": MOTOR_MODELS[motor_id],
        "host_id": HOST_ID,
        "state": state_dict(motor),
        "params": {},
    }
    for param_id, kind, name in PARAMS:
        try:
            result["params"][name] = {
                "id": f"0x{param_id:04X}",
                "type": kind,
                "value": read_param(motor, param_id, kind),
            }
        except Exception as exc:
            result["params"][name] = {"id": f"0x{param_id:04X}", "type": kind, "error": str(exc)}
    try:
        fault, warning = motor.robstride_get_fault_report()
        result["fault_raw"] = fault
        result["warning_raw"] = warning
    except Exception as exc:
        result["fault_error"] = str(exc)
    return result


def open_motors(channel: str, only_ids: tuple[int, ...]) -> tuple[Any, dict[int, Any]]:
    Controller, _CallError, _MotorBridgeError = import_motorbridge()
    try:
        controller = Controller(channel)
    except Exception as exc:
        die(f"无法打开 SocketCAN 通道 {channel!r}：{exc}\n请先确认 ip link 中存在并已拉起该通道。")
    motors: dict[int, Any] = {}
    try:
        for motor_id in only_ids:
            motor = controller.add_robstride_motor(motor_id, HOST_ID, MOTOR_MODELS[motor_id])
            motors[motor_id] = motor
    except Exception:
        for motor in motors.values():
            try:
                motor.close()
            except Exception:
                pass
        controller.close()
        raise
    return controller, motors


def assert_channel_is_explicitly_safe(channel: str, allow_platform_can: bool) -> None:
    sysfs = Path("/sys/class/net") / channel / "device"
    resolved = str(sysfs.resolve()) if sysfs.exists() else ""
    if "mttcan" in resolved and not allow_platform_can:
        die(
            f"{channel} 指向 Jetson 内置 MTT-CAN 控制器（{resolved}），"
            "不是已确认的 USB-CAN 接口。请先确认 USB-CAN 驱动实际创建的 canX；"
            "只有确实把机械臂接到 Jetson 板载 CAN 时，才加 --allow-platform-can。"
        )


def close_all(controller: Any, motors: dict[int, Any]) -> None:
    for motor in motors.values():
        try:
            motor.close()
        except Exception:
            pass
    try:
        controller.close()
    except Exception:
        pass


def scan(args: argparse.Namespace) -> int:
    assert_channel_is_explicitly_safe(args.channel, args.allow_platform_can)
    controller, motors = open_motors(args.channel, tuple(args.ids) if args.ids else MOTOR_IDS)
    found = 0
    try:
        print(f"通道：{args.channel}；RobStride host/feedback ID：0x{HOST_ID:02X}")
        for motor_id, motor in motors.items():
            try:
                device_id, responder_id = motor.robstride_ping_host_id(HOST_ID, args.timeout_ms)
                data = state_dict(motor)
                found += 1
                pos = "无反馈"
                if data is not None:
                    pos = f"{data['position_deg']:.2f}°，温度 {data['temperature_mos_c']:.1f}°C"
                print(
                    f"[在线] J{motor_id} {JOINT_NAMES[motor_id]:14s} "
                    f"{MOTOR_MODELS[motor_id]:5s} device=0x{device_id:02X} "
                    f"responder=0x{responder_id:02X}，位置 {pos}"
                )
            except Exception as exc:
                print(f"[离线] J{motor_id} {JOINT_NAMES[motor_id]:14s}：{exc}")
        print(f"发现 {found}/{len(motors)} 个目标电机。此步骤只读，未使能、未运动、未写参数。")
        return 0 if found == len(motors) else 1
    finally:
        close_all(controller, motors)


def snapshot(args: argparse.Namespace) -> int:
    assert_channel_is_explicitly_safe(args.channel, args.allow_platform_can)
    controller, motors = open_motors(args.channel, MOTOR_IDS)
    try:
        records = []
        for motor_id, motor in motors.items():
            try:
                motor.robstride_ping_host_id(HOST_ID, args.timeout_ms)
                records.append(motor_snapshot(motor, motor_id))
            except Exception as exc:
                records.append({"motor_id": motor_id, "joint": JOINT_NAMES[motor_id], "error": str(exc)})
        output = Path(args.out).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "tool": "b601_rs_calibrate.py",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "channel": args.channel,
            "host_id": HOST_ID,
            "platform": platform.platform(),
            "motors": records,
        }
        output.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n")
        print(f"已保存只读参数快照：{output}")
        return 0
    finally:
        close_all(controller, motors)


def calibrate(args: argparse.Namespace) -> int:
    if not args.confirm_mechanical_zero or not args.confirm_write:
        die("校零是写入电机的操作，必须同时提供 --confirm-mechanical-zero 和 --confirm-write。")
    if not args.backup:
        die("校零前必须用 --backup 指定参数备份文件。")
    backup = Path(args.backup).expanduser()
    if not backup.exists():
        die(f"找不到备份文件 {backup}。先运行 snapshot。")
    try:
        backup_doc = json.loads(backup.read_text())
        backup_time = datetime.fromisoformat(backup_doc["created_utc"])
        age_s = (datetime.now(timezone.utc) - backup_time).total_seconds()
        rows = {int(row["motor_id"]): row for row in backup_doc["motors"]}
    except Exception as exc:
        die(f"备份文件格式无效：{exc}")
    if backup_doc.get("channel") != args.channel or set(rows) != set(MOTOR_IDS):
        die("备份必须来自同一 CAN 通道且包含 J1-J7 的完整 snapshot。")
    if age_s < 0 or age_s > 600:
        die("备份需在最近 10 分钟内生成；重新运行 snapshot 后再校零。")
    if any(row.get("fault_raw", 1) != 0 or row.get("warning_raw", 1) != 0 for row in rows.values()):
        die("备份中存在电机故障或警告；请排除问题后重新 snapshot。")
    assert_channel_is_explicitly_safe(args.channel, args.allow_platform_can)

    print("即将对 B601-RS 的 7 个 RobStride 电机写入零点并保存参数。")
    print("请确认：朋友已停止所有控制；机械臂处于 B601-RS 手册规定的机械零位；夹爪处于规定零位。")
    print("当前静止放置位置、随意中间位置都不能直接当零点。")
    try:
        typed = input("输入大写 CALIBRATE 继续：").strip()
    except EOFError:
        print("未收到交互确认，已取消，未写入任何参数。")
        return 0
    if typed != "CALIBRATE":
        print("已取消，未写入任何参数。")
        return 0

    controller, motors = open_motors(args.channel, MOTOR_IDS)
    done: list[int] = []
    try:
        # Verify every motor responds before writing any one of them.
        for motor_id, motor in motors.items():
            device_id, responder_id = motor.robstride_ping_host_id(HOST_ID, args.timeout_ms)
            if device_id != motor_id or responder_id not in RESPONDER_IDS:
                raise RuntimeError(
                    f"J{motor_id} ID 不匹配：收到 device=0x{device_id:02X}, "
                    f"responder=0x{responder_id:02X}（允许 0xFD/0xFE/0xFF）"
                )
            fault, warning = motor.robstride_get_fault_report()
            if fault != 0:
                raise RuntimeError(f"J{motor_id} 存在故障，停止写入（fault=0x{fault:X}）")
            if warning:
                raise RuntimeError(f"J{motor_id} 存在 warning=0x{warning:X}，停止写入并先检查电机状态")
        print("7 个电机均已响应，开始逐个校零。")
        for motor_id, motor in motors.items():
            print(f"J{motor_id} {JOINT_NAMES[motor_id]}：失能 -> 设置零点 -> 保存参数")
            motor.disable()
            time.sleep(0.08)
            motor.set_zero_position()
            time.sleep(0.12)
            motor.store_parameters()
            time.sleep(0.12)
            done.append(motor_id)
        print(f"已完成 J{done}。建议重新运行 snapshot 做校零后的只读复核。")
        return 0
    except Exception as exc:
        print(f"校零在 J{done} 之后中止：{exc}", file=sys.stderr)
        return 1
    finally:
        close_all(controller, motors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seeed B601-RS RobStride 安全扫描/校零工具")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--channel", required=True, help="明确指定 CAN 通道，例如 can1")
    common.add_argument(
        "--allow-platform-can",
        action="store_true",
        help="允许使用 Jetson 内置 MTT-CAN；只有机械臂确实接在板载 CAN 时使用",
    )
    common.add_argument("--timeout-ms", type=int, default=700, help="单次响应超时，默认 700 ms")

    p_scan = sub.add_parser("scan", parents=[common], help="只读扫描目标电机")
    p_scan.add_argument("--ids", nargs="+", type=parse_id, help="只扫描指定 ID，例如 1 2 3")
    p_scan.set_defaults(func=scan)

    p_snapshot = sub.add_parser("snapshot", parents=[common], help="读取参数并保存 JSON，不写入")
    p_snapshot.add_argument("--out", required=True, help="输出 JSON 路径")
    p_snapshot.set_defaults(func=snapshot)

    p_cal = sub.add_parser("calibrate", parents=[common], help="写入机械零点并保存参数")
    p_cal.add_argument("--backup", required=True, help="先前 snapshot 生成的 JSON 备份")
    p_cal.add_argument("--confirm-mechanical-zero", action="store_true")
    p_cal.add_argument("--confirm-write", action="store_true")
    p_cal.set_defaults(func=calibrate)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))
