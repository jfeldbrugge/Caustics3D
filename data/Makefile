exec := cargo +nightly run --release --
archive := test.hdf5

.PHONY: all clean

all: a3.log

a3.log: eigen.log
	$(exec) a3 -n s2 $(archive) 2> $@

eigen.log: gadget-ics.log
	$(exec) eigen -s 2.0 -n s2 $(archive) 2> $@

gadget-ics.log: snapshot_ics_000.hdf5
	$(exec) gadget-ics $^ -o $(archive) 2> $@

clean:
	rm *.log

