for f in source_directory/*
        do echo "Prosessing $f file..."
        split --lines 10000 --numeric-suffixes --suffix-length=3 $f destination_directory/prefix_
done
