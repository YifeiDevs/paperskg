from server import mcp
import logging

# Configure logging to work with MCP system
file_handler = logging.FileHandler('paperkg.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s [%(levelname)s] [%(name)s.%(funcName)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
))

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)

logging.info("Starting paperkg MCP server initialization")

if __name__ == '__main__':
    try:
        logging.info("MCP server starting with stdio transport")
        mcp.run("stdio")
        logging.info("MCP server shutdown completed")
    except Exception as e:
        logging.error(f"MCP server failed to start: {e}")
        raise