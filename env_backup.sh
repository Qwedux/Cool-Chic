#!/bin/bash

BACKUP_DIR="${HOME}/conda_env_backup"
MY_TIME_FORMAT='%Y-%m-%d_%H-%M-%S'

NOW=$(date "+${MY_TIME_FORMAT}")
[[ ! -d "${BACKUP_DIR}" ]] && mkdir "${BACKUP_DIR}"
ENVS=$(conda env list |grep '^\w' |cut -d' ' -f1)
for env in $ENVS; do
    echo "Exporting ${env} to ${BACKUP_DIR}/${env}_${NOW}.yml"
    conda env export --name "${env}"> "${BACKUP_DIR}/${env}_${NOW}.yml"
done