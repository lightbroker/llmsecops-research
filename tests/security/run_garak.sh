export GARAK_PROBE='phrasing.FutureTense'


garak -vvv \
    --config ./garak.config.yml \
    --generator_option_file ./garak.rest.llm.json \
    --model_type=rest \
    --parallel_attempts 4 \
    --probes $GARAK_PROBE | tee garak_logs__$GARAK_PROBE.txt