import torch
from strats import STraTSNetwork
from coformer import CompatibleTransformer

def test_models():
    print("Iniciando validación de Arquitecturas Baseline...\n")
    B = 2    # Batch Size
    S = 100  # Samples irregulares (longitud de secuencia variable aplastada)
    num_features = 10

    # Dummy inputs
    times = torch.rand(B, S) * 100
    feature_ids = torch.randint(0, num_features, (B, S))
    values = torch.randn(B, S)
    mask = torch.ones(B, S, dtype=torch.bool)
    # Introducir un poco de padding
    mask[0, 80:] = False
    
    print("--- Probando STraTSNetwork ---")
    strats = STraTSNetwork(num_features=num_features, d_model=64, num_classes=1)
    out_strats = strats(times, feature_ids, values, mask)
    print(f"STraTS Output Shape: {out_strats.shape} (Esperado: [{B}, 1])\n")
    
    print("--- Probando Compatible Transformer (CoFormer) ---")
    coformer = CompatibleTransformer(num_variates=num_features, d_model=64, d_var=16, d_time=64, n_heads=4, n_layers=2)
    out_coformer = coformer(times, feature_ids, values, mask)
    print(f"CoFormer Output Shape: {out_coformer.shape} (Esperado: [{B}, 1])\n")
    
    print("Todas las pruebas pasaron correctamente!")

if __name__ == "__main__":
    test_models()
