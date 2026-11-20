# Windows Codex 后续提示词

我已从服务器下载`b601-pi05-deployment-v1.zip`及SHA256。请在Windows完成以下工作，不做机器人调优：

1. 校验ZIP的SHA256后解压到独立项目目录，保留原下载包。原ZIP是历史导出，当前Git版本的六轴速度上限修订以源码和`FILE_MANIFEST.json`为准。
2. 阅读README、docs/BASELINE.md、docs/DEPLOYMENT.md和THIRD_PARTY_NOTICES.md，运行`python scripts/verify_source.py`，只做静态检查，不import或运行Jetson控制器/云端模型服务。
3. 当前版本按作者确认将Pi0.5的`speed_deg_s`设为`null`，J1–J6（含J2）不设独立速度上限；不要恢复原ZIP中的5°/s。没有新授权时，不继续改运动参数、算法、单位转换、控制保护或模型代码。完善我的GitHub项目展示可以改文档，但准确保留结果和局限，不捏造成功率或演示视频。
4. 检查没有个人SSH地址/密码/令牌/权重/SDK二进制/全量数据。环境示例填写占位符，不上传下载服务器用到的私人连接信息。第三方署名与许可证保留，不把上游Pi0.5说成我从零开发。
5. 初始化本地Git仓库，配置Git身份前先检查已有用户配置，不编造姓名邮箱；提交当前整理版。若我已有目标GitHub仓库就使用它；目标未明确时只问公开还是私有及仓库名。使用现有GitHub登录，未登录时提示我完成正常认证，不索取访问令牌。
6. 我确认目标仓库后上传，并给我仓库链接。不要把SSH下载凭据写进Git历史；提交前先检查暂存文件。

项目运行对象是Jetson Linux和云端GPU，Windows用于阅读、静态校验和Git管理。完整权重不在包中，这不是开箱可运行的独立Pi0.5模型。
