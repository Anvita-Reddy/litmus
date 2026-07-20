from data import load_adult
from model import train_and_score
from shortcuts import inject_shortcut


def main():
    X, y = load_adult()
    train_and_score(X, y, "no shortcut")
    train_and_score(inject_shortcut(X, y), y, "with shortcut")


if __name__ == "__main__":
    main()