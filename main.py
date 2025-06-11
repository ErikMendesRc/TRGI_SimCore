import sys
import matplotlib
from gui.interactive_sim_mpl import TRGIInteractiveSim


def main():
    print(f"Starting TRGI_SimCore — backend: {matplotlib.get_backend()}")
    cfg = sys.argv[1] if len(sys.argv) > 1 else "config/default_params.json"
    TRGIInteractiveSim(config_path=cfg).run()
    print("TRGI_SimCore finished.")


if __name__ == "__main__":
    main()
