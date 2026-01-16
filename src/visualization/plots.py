import os
import matplotlib.pyplot as plt
from src.config.settings import OUTPUT_DIR

def plot_probability_over_time(results):
    """
    Plot collision probability vs time for each debris.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    debris_ids = sorted(set(r["debris_id"] for r in results))

    plt.figure(figsize=(10, 6))

    for d_id in debris_ids:
        times = [r["step"] for r in results if r["debris_id"] == d_id]
        probs = [r["probability"] for r in results if r["debris_id"] == d_id]

        if max(probs) > 0:
            plt.plot(times, probs, label=d_id)

    plt.xlabel("Time Step")
    plt.ylabel("Collision Probability")
    plt.title("Collision Probability Over Time")

    if len(debris_ids) <= 10:
        plt.legend()
    else:
        plt.legend(fontsize=8, ncol=2)

    save_path = os.path.join(OUTPUT_DIR, "probability_over_time.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Saved: {save_path}")


def plot_risk_ranking(results):
    """
    Plot max risk per debris (ranking).
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    risk_map = {}
    for r in results:
        d = r["debris_id"]
        risk_map[d] = max(risk_map.get(d, 0), r["risk"])

    debris = list(risk_map.keys())
    risks = [risk_map[d] for d in debris]

    plt.figure(figsize=(8, 5))
    plt.bar(debris, risks)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Max Risk Score")
    plt.title("Debris Risk Ranking")

    save_path = os.path.join(OUTPUT_DIR, "risk_ranking.png")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"[OK] Saved: {save_path}")
