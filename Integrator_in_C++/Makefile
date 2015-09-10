integrators = euler.o
equations = duffing.o
objects = duffing_solve.o $(integrators) $(equations)

CXXFLAGS = -g -Wall

all: duffing_solve

duffing_solve : $(objects)
	$(CXX) -o $@ $^

clean:
	$(RM) *.o
	$(RM) .depend

depend:
	$(CXX) -MM $(CXXFLAGS) *.cc > .depend

-include .depend
