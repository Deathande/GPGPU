
if [ $# -lt 2 ]
then
  echo "too few arguments"
else
  make
  if [ $? -eq 0 ]
  then
    ./create 'test.dat' $1 && ./hist 'test.dat' $1 $2
  fi
fi


