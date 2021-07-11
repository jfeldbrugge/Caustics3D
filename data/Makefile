exec := cargo +nightly run --release --
archive := test.hdf5

.PHONY: all clean build

all: a3.log

build:
	cargo +nightly build

a3.log: eigen.log
	$(exec) a3 -n s5 $(archive) --dump-obj a3.obj 2> $@

eigen.log: gadget-ics.log
	$(exec) eigen -s 5.0 -n s5 $(archive) 2> $@

gadget-ics.log: snapshot_ics_000.hdf5
	$(exec) gadget-ics $^ -o $(archive) 2> $@

clean:
	rm *.log

