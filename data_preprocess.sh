#!/bin/bash

DATASET=${1}
DATA_TYPE=${2}
DATA_ROOT=${3}
SAVE_NAME=${4}

if [ -d "$DATA_ROOT$SAVE_NAME" ]; then
  echo "directory $DATA_ROOT$SAVE_NAME already exist."
else
  mkdir $DATA_ROOT$SAVE_NAME
fi

if [ ${DATASET} = 'fb15k237' ];
then
  export MAX_ENT_LEN=18
  export MAX_REL_LEN=38
  export MAX_SEQ_LEN=192
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} train ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} test ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} dev ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
elif [ ${DATASET} = 'fb15k237-roberta' ];
then
  export MAX_ENT_LEN=21
  export MAX_REL_LEN=23
  export MAX_SEQ_LEN=187
  python ./data/preprocess_neg.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} train ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess_neg.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} test ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess_neg.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} dev ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
else
  export MAX_ENT_LEN=17
  export MAX_REL_LEN=5
  export MAX_SEQ_LEN=192
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} train ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} test ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
  python ./data/preprocess.py ${MAX_ENT_LEN} ${MAX_REL_LEN} ${MAX_SEQ_LEN} dev ${DATA_TYPE} ${DATA_ROOT} $DATA_ROOT$SAVE_NAME
fi






