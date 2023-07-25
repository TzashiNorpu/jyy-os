CFLAGS = -ggdb -Wall

.PHONY:a.out debug clean

a.out: hello.c
	gcc $(CFLAGS) hello.c

debug: a.out
	gdb -x init.gdb

clean:
	rm -rf a.out