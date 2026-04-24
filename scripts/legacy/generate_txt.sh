#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )"/.. && pwd )"
FOLDERS=("src" "configs" "scripts")
OUTPUT_FILE="$REPO_ROOT/ALL_FILES_CONTENT.txt"

> "$OUTPUT_FILE"
echo "Generando $OUTPUT_FILE ..."

for FOLDER in "${FOLDERS[@]}"; do
    FOLDER_PATH="$REPO_ROOT/$FOLDER"

    find "$FOLDER_PATH" -type f \( -name "*.py" -o -name "*.yaml" -o -name "*.sh" -o -name "*.ps1" \) | sort | while read -r FILE; do
        echo "Procesando $FILE ..."

        REL_PATH="${FILE#${REPO_ROOT}/}"

        echo "$REL_PATH" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"
        cat "$FILE" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo '```' >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
    done
done

echo "Generación completada: $OUTPUT_FILE"
