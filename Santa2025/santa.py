import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    x = np.linspace(0, np.pi, 100)
    y = np.sin(x)

    plt.plot(x, y)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
