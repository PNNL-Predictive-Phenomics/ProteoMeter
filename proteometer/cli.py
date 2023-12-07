"""Console script for proteometer."""

import fire


def help() -> None:
    print("proteometer")
    print("=" * len("proteometer"))
    print("ProteoMeter: A Comprehensive Python Library for Proteom")


def main() -> None:
    fire.Fire({"help": help})


if __name__ == "__main__":
    main()  # pragma: no cover
