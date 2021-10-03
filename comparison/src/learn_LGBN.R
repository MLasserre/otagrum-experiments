library(bnlearn)

args = commandArgs(trailingOnly=TRUE)
if(length(args) != 4){
    stop("Le script attend en entrée: 1) le chemin du fichier de données, 2) le chemin du
          fichier où écrire la structure apprise, 3) le nombre de restarts et 4) le nombre
          maximal de parents que peut avoir un noeud.",
         call.=FALSE)
}

input_file_name = args[1]
output_file_name = args[2]
r = as.integer(args[3])
mp = as.integer(args[4])

donnees <- read.csv(file=input_file_name, header=TRUE, sep=",")
cat("Le fichier", input_file_name, "a été correctement chargé.\n")

print("Apprentissage de la structure...")
bn = hc(donnees, score="bge", restart=r, maxp=mp, debug=FALSE)

# Ecriture du fichier dot contenant le graphe appris
cat("digraph {\n", file=output_file_name)
write.table(bn$arcs,
            file=output_file_name,
            quote=FALSE,
            sep="->",
            row.names=FALSE,
            col.names=FALSE,
            eol=";\n",
            append=TRUE)
cat("}", file=output_file_name, append=TRUE)
