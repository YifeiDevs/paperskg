from server import mcp
import logging

file_handler = logging.FileHandler('paperkg.log', mode='a')
logger = logging.getLogger()
logger.addHandler(file_handler)
logging.info("paperkg MCP server initialized")

if __name__ == '__main__':
    # 在项目目录, 按下 Ctrl + ` 打开 powershell 终端:
    # uv init .
    # uv add "mcp[cli]"
    mcp.run("stdio")