input_dir_name=${1:-"../results/structures/"}
output_dir_name=${2:-"../figures/structures/"}

mkdir -p "$output_dir_name"

for file in "${input_dir_name}"/*.dot
do
    filename=$(basename -- $file .dot)
    echo "Processing ${filename}"
    #echo "${output_dir_name}/${filename%.*}.pdf"
    dot -Tpdf $file -o "${output_dir_name}/${filename%.*}.pdf"
done
