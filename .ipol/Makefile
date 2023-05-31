ddl.json: ddl.txt
	cat $^ | sed 's/^#.*//' | awk NF | 2json > $@

clean:
	$(RM) ddl.json
