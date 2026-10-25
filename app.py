
import base64
import io
import math
import traceback

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request
from scipy.optimize import brentq

app = Flask(__name__)

# Constantes d'Antoine : pression de vapeur en mmHg,
# température en °C.
# Formule : log10(Psat) = A - B / (T + C)
ANTOINE = {
    "water": (8.07131, 1730.63, 233.426),
    "ethanol": (8.20417, 1642.89, 230.300),
}

MMHG_PER_BAR = 750.061683


def psat_bar(component, temperature_c):
    """Pression de vapeur saturante en bar."""
    A, B, C = ANTOINE[component]
    p_mmhg = 10 ** (A - B / (temperature_c + C))
    return p_mmhg / MMHG_PER_BAR


def bubble_temperature(x_ethanol, pressure_bar):
    """Température de bulle selon la loi de Raoult."""
    x = float(np.clip(x_ethanol, 0.0, 1.0))

    def objective(T):
        return (
            (1 - x) * psat_bar("water", T)
            + x * psat_bar("ethanol", T)
            - pressure_bar
        )

    return brentq(objective, 0.1, 120.0)


def vapor_ethanol_fraction(x_ethanol, pressure_bar):
    """Fraction molaire d'éthanol dans la vapeur à l'équilibre."""
    T = bubble_temperature(x_ethanol, pressure_bar)
    p_ethanol = psat_bar("ethanol", T)
    return x_ethanol * p_ethanol / pressure_bar


def dew_temperature(y_ethanol, pressure_bar):
    """Température de rosée selon la loi de Raoult."""
    y = float(np.clip(y_ethanol, 0.0, 1.0))

    def objective(T):
        pw = psat_bar("water", T)
        pe = psat_bar("ethanol", T)
        return (1 - y) * pressure_bar / pw + y * pressure_bar / pe - 1

    return brentq(objective, 0.1, 120.0)


def fig_to_base64(fig):
    """Convertit une figure Matplotlib en image intégrable dans HTML."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def make_txy_plot(pressure_bar):
    """Génère le diagramme binaire T-x-y."""
    x_values = np.linspace(0, 1, 101)
    y_values = np.linspace(0, 1, 101)

    bubble_values = [
        bubble_temperature(x, pressure_bar) for x in x_values
    ]
    dew_values = [
        dew_temperature(y, pressure_bar) for y in y_values
    ]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x_values, bubble_values, label="Courbe d'ébullition")
    ax.plot(y_values, dew_values, label="Courbe de rosée")

    ax.set_xlabel("Fraction molaire d'éthanol (x ou y)")
    ax.set_ylabel("Température (°C)")
    ax.set_title(f"Diagramme T-x-y eau–éthanol à {pressure_bar:.3f} bar")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xlim(0, 1)

    return fig_to_base64(fig)


def make_mccabe_plot(xf, xd, xb, reflux, q, pressure_bar):
    """Génère un diagramme de McCabe–Thiele simplifié."""
    if not (0 < xb < xf < xd < 1):
        raise ValueError(
            "Il faut respecter 0 < xB < xF < xD < 1."
        )

    if reflux <= 0:
        raise ValueError("Le taux de reflux doit être positif.")

    x_eq = np.linspace(0, 1, 501)
    y_eq = np.array([
        vapor_ethanol_fraction(x, pressure_bar) for x in x_eq
    ])

    # Droite opératoire de rectification.
    def y_rect(x):
        return reflux / (reflux + 1) * x + xd / (reflux + 1)

    # Droite q : alimentation.
    if abs(q - 1.0) < 1e-8:
        x_intersect = xf
        y_intersect = y_rect(x_intersect)
        q_slope = None
    else:
        q_slope = q / (q - 1)
        q_intercept = -xf / (q - 1)
        denominator = q_slope - reflux / (reflux + 1)

        if abs(denominator) < 1e-10:
            raise ValueError(
                "Les droites d'alimentation et de rectification "
                "sont presque parallèles."
            )

        x_intersect = (
            xd / (reflux + 1) - q_intercept
        ) / denominator
        y_intersect = y_rect(x_intersect)

    # Droite opératoire d'épuisement passant par (xB,xB)
    # et par le point d'intersection.
    if abs(x_intersect - xb) < 1e-10:
        raise ValueError(
            "Impossible de construire la droite d'épuisement "
            "avec ces paramètres."
        )

    stripping_slope = (y_intersect - xb) / (x_intersect - xb)

    def y_strip(x):
        return stripping_slope * (x - xb) + xb

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(x_eq, y_eq, label="Équilibre liquide-vapeur")
    ax.plot([0, 1], [0, 1], "--", label="Diagonale y = x")

    x_rect = np.linspace(max(0, min(xf, xd)), xd, 100)
    ax.plot(x_rect, [y_rect(x) for x in x_rect],
            label="Droite de rectification")

    x_strip = np.linspace(xb, min(1, max(xf, xb)), 100)
    ax.plot(x_strip, [y_strip(x) for x in x_strip],
            label="Droite d'épuisement")

    # Marche théorique entre l'équilibre et les droites opératoires.
    current_x = xd
    current_y = xd
    stage_count = 0

    for _ in range(100):
        # Horizontal : de la droite opératoire vers l'équilibre.
        if current_y < np.min(y_eq) or current_y > np.max(y_eq):
            break

        next_x = float(np.interp(current_y, y_eq, x_eq))

        ax.plot([current_x, next_x], [current_y, current_y], "k-", lw=1)
        ax.plot([next_x, next_x], [current_y, next_x], "k-", lw=1)

        stage_count += 1
        current_x = next_x
        current_y = next_x

        if current_x <= xb:
            break

        operating_line = y_rect(current_x) if current_x >= xf else y_strip(current_x)
        ax.plot([current_x, current_x], [current_y, operating_line],
                color="gray", lw=0.8)

        current_y = operating_line

    ax.set_xlabel("Fraction molaire d'éthanol dans le liquide x")
    ax.set_ylabel("Fraction molaire d'éthanol dans la vapeur y")
    ax.set_title("Diagramme de McCabe–Thiele simplifié")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    return fig_to_base64(fig), stage_count


def parse_float(form, name, default):
    """Lit un nombre du formulaire ou utilise la valeur par défaut."""
    value = form.get(name, str(default)).strip()
    return float(value) if value else float(default)


@app.route("/", methods=["GET", "POST"])
def index():
    error = None
    txy_image = None
    mccabe_image = None
    stage_count = None
    result_flash = []

    # Valeurs initiales de la colonne
    xf = 0.207
    xd = 0.80
    xb = 0.05
    reflux = 2.0
    q = 0.5
    pressure_bar = 1.01325

    if request.method == "POST":
        try:
            action = request.form.get("action", "distillation")

            xf = parse_float(request.form, "xf", xf)
            xd = parse_float(request.form, "xd", xd)
            xb = parse_float(request.form, "xb", xb)
            reflux = parse_float(request.form, "reflux", reflux)
            q = parse_float(request.form, "q", q)
            pressure_bar = parse_float(
                request.form, "pressure_column", pressure_bar
            )

            if pressure_bar <= 0:
                raise ValueError("La pression doit être positive.")

            if action == "distillation":
                txy_image = make_txy_plot(pressure_bar)
                mccabe_image, stage_count = make_mccabe_plot(
                    xf, xd, xb, reflux, q, pressure_bar
                )

            elif action == "flash":
                xa = parse_float(request.form, "xa", 0.6)
                xe = parse_float(request.form, "xe", 0.4)
                temperature = parse_float(
                    request.form, "temperature", 40
                )
                flash_pressure = parse_float(
                    request.form, "pressure", 1.01325
                )
                mass_flow = parse_float(
                    request.form, "mass_flow", 30
                )
                temp_step = parse_float(request.form, "step", 5)

                if xa < 0 or xe < 0 or abs(xa + xe - 1) > 0.01:
                    raise ValueError(
                        "Les fractions molaires d'eau et d'éthanol "
                        "doivent être positives et leur somme doit être 1."
                    )

                if flash_pressure <= 0 or temp_step <= 0 or mass_flow < 0:
                    raise ValueError(
                        "Vérifie la pression, le pas de température "
                        "et le débit massique."
                    )

                # Tableau de volatilité relative simplifié.
                temperatures = np.arange(
                    temperature, temperature + 5 * temp_step + 0.001,
                    temp_step
                )

                z_ethanol = xe
                for T in temperatures:
                    try:
                        K_water = psat_bar("water", T) / flash_pressure
                        K_ethanol = psat_bar("ethanol", T) / flash_pressure
                        z_water = xa

                        def rachford_rice(beta):
                            return (
                                z_water * (K_water - 1)
                                / (1 + beta * (K_water - 1))
                                + z_ethanol * (K_ethanol - 1)
                                / (1 + beta * (K_ethanol - 1))
                            )

                        f0 = rachford_rice(0)
                        f1 = rachford_rice(1)

                        if f0 * f1 <= 0:
                            beta = brentq(rachford_rice, 0, 1)
                        else:
                            beta = 0.0 if f0 < 0 else 1.0

                        xw_den = 1 + beta * (K_water - 1)
                        xe_den = 1 + beta * (K_ethanol - 1)
                        x_water = z_water / xw_den
                        x_ethanol = z_ethanol / xe_den

                        y_water = K_water * x_water
                        y_ethanol = K_ethanol * x_ethanol

                        result_flash.append({
                            "temperature": round(float(T), 2),
                            "beta": round(float(beta), 4),
                            "x_water": round(float(x_water), 4),
                            "x_ethanol": round(float(x_ethanol), 4),
                            "y_water": round(float(y_water), 4),
                            "y_ethanol": round(float(y_ethanol), 4),
                        })
                    except Exception as exc:
                        result_flash.append({
                            "temperature": round(float(T), 2),
                            "error": str(exc),
                        })

        except Exception as exc:
            error = str(exc)
            app.logger.error("Erreur de calcul : %s", traceback.format_exc())

    return render_template(
        "index.html",
        error=error,
        txy_image=txy_image,
        mccabe_image=mccabe_image,
        stage_count=stage_count,
        result_flash=result_flash,
        xf=xf,
        xd=xd,
        xb=xb,
        reflux=reflux,
        q=q,
        pressure_bar=pressure_bar,
    )


if __name__ == "__main__":
    app.run(debug=True)

