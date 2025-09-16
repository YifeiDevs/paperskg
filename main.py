from server import mcp

if __name__ == '__main__':
    # 在项目目录, 按下 Ctrl + ` 打开 powershell 终端:
    # uv init .
    # uv add "mcp[cli]"
    mcp.run("stdio")