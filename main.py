import matplotlib

# Tente diferentes backends se houver problemas de renderização ou interatividade
# Comuns: 'TkAgg', 'Qt5Agg', 'QtAgg', 'WXAgg'
# matplotlib.use('TkAgg') # Exemplo

from gui.interactive_sim_mpl import TRGIInteractiveSim
import sys

def main():
    print("Starting TRGI_SimCore...")
    print("Using Matplotlib backend: {matplotlib.get_backend()}")

    # Verificar se um arquivo de configuração foi passado como argumento
    config_file = 'config/default_params.json'
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
        print("Attempting to load custom config: {config_file}")

    app = TRGIInteractiveSim(config_path=config_file)
    app.run()
    print("TRGI_SimCore finished.")

if __name__ == '__main__':
    main()