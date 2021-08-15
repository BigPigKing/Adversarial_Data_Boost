textattack augment --input-csv $1_test.csv --output-csv $1_stack_eda.csv --input-column sentence --recipe eda --pct-words-to-swap .1 --transformations-per-example 5 --exclude-original --overwrite
textattack augment --input-csv $1_test.csv --output-csv $1_eda.csv --input-column sentence --recipe eda --pct-words-to-swap .1 --transformations-per-example 4 --exclude-original --overwrite
textattack augment --input-csv $1_test.csv --output-csv $1_embedding.csv --input-column sentence --recipe embedding --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite
textattack augment --input-csv $1_test.csv --output-csv $1_clare.csv --input-column sentence --recipe clare --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite
textattack augment --input-csv $1_test.csv --output-csv $1_checklist.csv --input-column sentence --recipe checklist --pct-words-to-swap .1 --transformations-per-example 2 --exclude-original --overwrite
