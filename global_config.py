from pathlib import Path

base_path = Path(__file__).parent
data_path = base_path / 'data'

if __name__ == '__main__':
    print(base_path)