import sys
from client import LionOptimizer

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

def run():
    print(">>> Demonstrating Lion (EvoLved Sign Momentum) Optimizer...")
    lion = LionOptimizer(lr=0.05, weight_decay=0.01)
    param = 10.0

    for step in range(1, 11):
        param = lion.step(param, grad=param)
        if step % 2 == 0:
            print(f"Step {step:2d}: param = {param:.6f}")

    assert param < 10.0
    print("[PASS] Lion Optimizer verified.")

if __name__ == "__main__":
    run()
