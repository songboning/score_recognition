#!/bin/bash

ls *pdf > _pdf_list.txt
cat _pdf_list.txt | while read PDF
do
  echo $PDF
  convert -density 100 -quality 100 "$PDF" "$PDF.jpg"
done
rm _pdf_list.txt
