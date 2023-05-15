grep -v '^#' benchmarks.txt | grep -v -e '^$' | while read -r file ; do
   echo "==================== TESTING: $file ===================="
   python3 clusterize.py --cluster-config-path $file
done
