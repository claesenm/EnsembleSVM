doxygen.cfg.stamp:
	$(DOXYGEN) doxygen.cfg
	touch doxygen.cfg.stamp

#	echo Timestamp > doxygen.cfg.stamp

CLEANFILES = doxygen.cfg.stamp

all-local: doxygen.cfg.stamp
clean-local:
	rm -rf doxygen.cfg.stamp
	rm -rf latex
	rm -rf html