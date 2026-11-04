import logging

import uvicorn

from graphai.config import get_settings


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        "graphai.app:app",
        host=settings.agent_host,
        port=settings.agent_port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
