while getopts "p:n:r:s:f:e:" opt
do
   case "$opt" in
      p ) path="$OPTARG" ;;
      n ) nprocesses="$OPTARG" ;;
      r ) release_or_debug="$OPTARG" ;;
      s ) skip_compile="$OPTARG" ;;
      f ) parameter_file="$OPTARG" ;;
      e ) execute="$OPTARG" ;;
   esac
done

path="${path:-build}"
nprocesses="${nprocesses:-8}"
release_or_debug="${release_or_debug:-release}"
skip_compile="${skip_compile:-false}"
execute="${execute:-true}"

echo "Path to build: ${path}"
echo "N threads: ${nprocesses}"
echo "Release or debug: ${release_or_debug}"
echo "Not compile: ${skip_compile}"

if [ "$skip_compile" = "true" ]; then
   echo "Skipping compiling"
   cd "$path"
else
   echo "Start compiling"
   rm -rf "$path"
   mkdir "$path"
   cd "$path"
   cmake ..
   make "$release_or_debug"
   make
fi

if [ "$execute" = "true" ]; then
   echo "Input file: ../${parameter_file}"
   mpirun --oversubscribe -n "$nprocesses" ./main "../${parameter_file}"
else
   echo "Finished"
fi

