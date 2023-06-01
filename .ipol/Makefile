ddl.json: ddl.txt
	@cat $^ | sed 's/^#.*//' | awk NF | 2json #> $@
	@#cat $@

clean:
	$(RM) ddl.json
