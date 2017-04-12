#include <stdio.h>
#include <stdlib.h>

int main(void) {
	int i = 1990;
	int j;
	char command[200];

	for (;i<2017;i++) {
		for (j=1;j<13;j++){
			sprintf(command, "wget ftp://www.ncedc.org/pub/catalogs/anss/%d/%d.%02d.cnss", i,i,j);
			system(command);
			printf("Got %d/%02d\n", i, j);
		}
	}
	return 0;
}
