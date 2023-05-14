grep -v '^#' benchmarks.txt | while read -r file ; do
   echo "==================== TESTING: $line ===================="
   python3 clusterize.py --cluster-config-path $line
done
