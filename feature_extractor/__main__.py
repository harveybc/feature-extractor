import sys
import feature_extractor.feature_extractor as fe

def main(args=None):
    """The main routine."""
    if args is None:
        args = sys.argv[1:]
    # you want your project to do.
    fe.run(args)

if __name__ == "__main__":
    main()
