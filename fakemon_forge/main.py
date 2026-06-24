from fakemon_forge.cli import parse_args, validate_args


def main():
    args = parse_args()
    validate_args(args)


if __name__ == "__main__":
    main()
