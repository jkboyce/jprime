
/************************************************************************/
/*   Jdeep.c                        by Jack Boyce        6/98           */
/*                                 jboyce@physics.berkeley.edu          */
/*                                                                      */
/*   This is a modification of the original J.c, optimized for speed.   */
/*   It is used to find "prime" (no subpatterns) async siteswaps.       */
/*   Basically it works by finding cycles in the state transition       */
/*   matrix.  Try the cases:                                            */
/*       jdeep 4 7 29      (0.4 sec on my slow machine)                 */
/*       jdeep 5 8 49      (85.0 sec)                                   */
/************************************************************************/

/*  This version finds the loops recursively  */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>


void die();


/*  num_states  --  This routine finds the number of allowed states for  */
/*                  a given number of balls and maximum throwing height. */

int num_states(int n, int h)
{
   int result = 1, num = h, denom = 1;

   if (h < (2*n))
      n = h - n;

   while (denom <= n) {
      result = (result * num) / denom;
      num--;
      denom++;
   }

   return result;
}


/*  gen_states  --  This recursively generates the possible states,    */
/*                  putting them into the state[][] array.             */

int gen_states(unsigned long *state, int num, int pos, int left, int h, int ns)
{
   if (left > (pos+1))
      return num;

   if (pos == 0) {
      if (left)
         state[num] |= 1L;
      else
         state[num] &= ~1L;

      if (num < (ns-1))
         state[num + 1] = state[num];
      return (num + 1);
   }

   state[num] &= ~(1L << pos);
   num = gen_states(state, num, pos-1, left, h, ns);
   if (left > 0) {
      state[num] |= 1L << pos;
      num = gen_states(state, num, pos-1, left-1, h, ns);
   }

   return num;
}


/*  gen_matrix  --  Once the states are found, this routine generates    */
/*                  the matrix giving the throws needed to go from each  */
/*                  state to every other (0 indicates no such throw).    */

void gen_matrix(int **matrix, int **throwval, unsigned long *state, int h,
      int ns, int nthrows)
{
   int i, j, k, thrownum;
   unsigned long temp, temp2;
   int found;

   for (i = 1; i <= ns; i++) {
      thrownum = 0;
      for (j = 0; j <= h; j++) {
         if (j == 0) {
            if (!(state[i] & 1L)) {
               temp = state[i] >> 1;
               found = 0;
               for (k = 1; k <= ns; k++) {
                  if (state[k] == temp) {
                     matrix[i][thrownum] = k;
                     throwval[i][thrownum++] = j;
                     found = 1;
                     break;
                  }
               }
               if (found == 0)
                  printf("Error:  Couldn't find state!\n");
            }
         } else {
            if (state[i] & 1L) {
               temp = (unsigned long)1L << (j-1);
               temp2 = (state[i] >> 1);
               if (!(temp2 & temp)) {
                  temp |= temp2;
                  found = 0;
                  for (k = 1; k <= ns; k++) {
                     if (state[k] == temp) {
                        matrix[i][thrownum] = k;
                        throwval[i][thrownum++] = j;
                        found = 1;
                        break;
                     }
                  }
                  if (found == 0)
                     printf("Error:  Couldn't find state!\n");
               }
            }
         }
      }
   }
}


/*  gen_patterns  --  The following functions actually generate the  */
/*                    juggling patterns, using the throwing matrix   */
/*                    created above.                                 */

void print_pattern(int *pattern, int pos, int start)
{
   int i, j;

   if (start == 1)
      printf("  ");
   else
      printf("* ");

   for (i = 0; i <= pos; i++) {
      j = pattern[i];
      if (j < 10)
         printf("%d", j);
      else
         printf("%c", j - 10 + 'A');
   }

   if (start != 1)
      printf(" *");

   printf("\n");

}

int gen_loops(int **matrix, int **throwval, unsigned long *state, int start,
      int n, int ns, int l, int numflag, int nthrows, int num, int *pattern,
      int *used, int from, int pos)
{
   int thrownum, to;
   int *matrixptr = matrix[from];
   int *throwptr = throwval[from];

   for (thrownum = 0; thrownum < nthrows; thrownum++) {
      to = *matrixptr++;
      if ((to >= start) && (used[to] == 0)) {
            /* are we finished? */
         if (to == start) {
            if ((pos >= l) || (l == -1)) {
               if (numflag != 2) {
                  pattern[pos] = throwptr[thrownum];
                  print_pattern(pattern, pos, start);
               }
               num++;
            }
         } else /* if ((pos < l) || (l == -1)) */ {	/* not finished */
            pattern[pos] = throwptr[thrownum];
            used[to] = 1;
            num = gen_loops(matrix, throwval, state, start,
                        n, ns, l, numflag, nthrows,
                        num, pattern, used, to, pos+1);
            used[to] = 0;
         }
      }
   }

   return num;
}

int gen_patterns(int **matrix, int **throwval, unsigned long *state,
      int *pattern, int *used, int n, int ns, int l, int numflag, int nthrows)
{
   int i, num = 0;

   for (i = 1; i <= ns; i++)
      num += gen_loops(matrix, throwval, state, i, n, ns, l, numflag, nthrows,
               0, pattern, used, i, 0);

   return num;
}


int main(int argc, char **argv)
{
   int **matrix, **throwval, *pattern, *used;
   unsigned long *state;
   int n, h, ns, l = -1, numflag = 0, np, nthrows, i, j;

   if (argc < 3) {
      printf("Usage: %s <# objects> <max. throw> [<min. length>] [-n[o]]\n",
        argv[0]);
      exit(0);
   }

   n = atoi(argv[1]);                    /* get the number of objects */
   if (n < 1) {
      printf("Must have at least 1 object\n");
      exit(0);
   }
   h = atoi(argv[2]);                    /* get the max. throw height */
   if (h < n) {
      printf("Max. throw height must equal or exceed number of objects\n");
      exit(0);
   }
   ns = num_states(n, h);			/* number of states */
   nthrows = h - n + 1;			/* max. number of throws from state */

   for (i = 3; i < argc; i++) {
      if (!strcmp(argv[i], "-n"))
         numflag = 1;
      else if (!strcmp(argv[i], "-no"))
         numflag = 2;
      else
         l = atoi(argv[i]) - 1;
   }

    /* Now allocate the memory space for the list of states,  */
    /* the throwing matrix, and other stuff.                  */

   if ((state = (unsigned long *)malloc((ns+2) * sizeof(unsigned long))) == 0)
      die();
   if ((matrix = (int **)malloc((ns+1) * sizeof(int *))) == 0)
      die();
   for (i = 0; i <= ns; i++)
      if ((matrix[i] = (int *)malloc(nthrows * sizeof(int))) == 0)
         die();
   if ((throwval = (int **)malloc((ns+1) * sizeof(int *))) == 0)
      die();
   for (i = 0; i <= ns; i++)
      if ((throwval[i] = (int *)malloc(nthrows * sizeof(int))) == 0)
         die();
   if ((pattern = (int *)malloc(ns * sizeof(int))) == 0)
      die();
   if ((used = (int *)malloc((ns+1) * sizeof(int))) == 0)
      die();

   for (i = 0; i < ns; i++)
      state[i] = 0L;
   (void)gen_states(state+1, 0, h - 1, n, h, ns); /* generate list of states */

   for (i = 0; i <= ns; i++)
      for (j = 0; j < nthrows; j++)
         matrix[i][j] = throwval[i][j] = 0;
   gen_matrix(matrix, throwval, state, h, ns, nthrows);

   for (i = 1; i <= ns; i++)
      used[i] = 0;

   np = gen_patterns(matrix, throwval, state, pattern, used,
            n, ns, l, numflag, nthrows);

   if (numflag != 0)
      printf("%d patterns found\n", np);

   free(used);
   free(pattern);
   for (i = 0; i <= ns; i++) {
      free(matrix[i]);
      free(throwval[i]);
   }
   free(matrix);
   free(throwval);
   free(state);

   return 0;
}


void die()
{
   printf("Insufficient memory\n");
   exit(0);
}
