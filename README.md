# Memoria

Este repositorio ahora tiene scripts separados por plataforma:

- Linux / WSL: `scripts/linux`
- Windows PowerShell: `scripts/windows`
- Scripts antiguos archivados: `scripts/delete`

Los scripts `.sh` originales quedaron archivados en `scripts/delete`.

## Preparacion del entorno

Activa tu entorno de Python antes de ejecutar cualquier script. En tu caso actual, por ejemplo:

```powershell
conda activate gpu
pip install -r requirements.txt
```

## Windows

Ejemplos desde PowerShell:

```powershell
.\scripts\windows\train_test1.ps1
.\scripts\windows\train_real_data1.ps1
.\scripts\windows\pretrain_real_data1.ps1
.\scripts\windows\finetune_real_data1.ps1
.\scripts\windows\run_real_data1.ps1 -Stage pretrain -ExperimentName pretrain_real_data1
.\scripts\windows\run_real_data1.ps1 -Stage finetune -ExperimentName exp_real_data1
.\scripts\windows\plot_real_data1.ps1
.\scripts\windows\plot_experiment_real_data1.ps1 -ExperimentName exp_real_data1 -MaxPoints 1000
.\scripts\windows\generate_txt.ps1
```

Si PowerShell bloquea la ejecucion de scripts, puedes habilitarlos solo para la sesion actual:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## Linux o WSL

Ejemplos desde Bash:

```bash
bash scripts/linux/train_test1.sh
bash scripts/linux/train_real_data1.sh
bash scripts/linux/pretrain_real_data1.sh
bash scripts/linux/finetune_real_data1.sh
bash scripts/linux/run_real_data1.sh pretrain pretrain_real_data1
bash scripts/linux/run_real_data1.sh finetune exp_real_data1
bash scripts/linux/plot_real_data1.sh
bash scripts/linux/plot_experiment_real_data1.sh exp_real_data1 1000
bash scripts/linux/generate_txt.sh
```

## Notas

- Todos los scripts configuran `PYTHONPATH` automaticamente apuntando a `src`.
- `generate_txt` dejo de depender de una ruta fija de Ubuntu y ahora resuelve el root del repo de forma dinamica.
