from data import load_adult
from model import train_and_score


def main():
    X, y = load_adult()
    print(f"Rows: {len(X):,}  Features: {X.shape[1]}")
    train_and_score(X, y, " ")
    print()


if __name__ == "__main__":
    main()