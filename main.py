from server import mcp
import logging

file_handler = logging.FileHandler('paperkg.log', mode='a')
logger = logging.getLogger()
logger.addHandler(file_handler)
logging.info("paperkg MCP server initialized")

if __name__ == '__main__':
    mcp.run("stdio")