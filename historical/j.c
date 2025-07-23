
/************************************************************************/
/*   J.c                        by Jack Boyce        11/90              */
/*                                 jboyce@tybalt.caltech.edu            */
/*                                                                      */
/*   This program finds all juggling patterns for a given number of     */
/*   balls and a given throwing height.  The state space approach is    */
/*   used in order to speed up computation.                             */
/************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void die();


int fact(int x)                 /* This just finds x! */
{
   int i, j = 1;

   if (x < 2)
      return 1;
   else
      for (i = 2; i <= x; i++)
         j *= i;
   return j;
}


/*  num_states  --  This routine finds the number of allowed states for  */
/*                  a given number of balls and maximum throwing height. */

int num_states(int n, int h)
{
   int ns;

   ns = fact(h - 2) / (fact(h - n) * fact(n - 2));         /* C(h-2, n-2) */
   if (h > 3)
      ns += fact(h - 3) / (fact(h - n - 1) * fact(n - 2)); /* C(h-3, n-2) */
   return ns;
}


/*  gen_states  --  This recursively generates the possible states,    */
/*                  putting them into the state[][] array.             */

int gen_states(int **state, int num, int pos, int n, int h)
{
   int i;

   if (pos == 0) {
      state[num][0] = 1;
      for (i = 0; i < n; i++)
         state[num + 1][i] = state[num][i];
      return (num + 1);
   }

   for (i = pos + 1; i <= (pos == 1 ? (h < 3 ? 2 : 3) : h); i++) {
      state[num][pos] = i;
      num = gen_states(state, num, pos - 1, n, i - 1);
   }

   return num;
}


/*  gen_matrix  --  Once the states are found, this routine generates    */
/*                  the matrix giving the throws needed to go from each  */
/*                  state to every other (0 indicates no such throw).    */

int matrix_element(int **state, int from, int to, int n)
/* generates matrix element */
{
   int i, j, throw = 0, temp;

   for (i = 0; i < n; i++) {
      temp = state[to][i] + 1;
      j = 1;
      while ((state[from][j] != temp) && (j < n))
         j++;
      if (j == n) {               /* no match in 'from' state */
         if (throw != 0)
            return 0;             /* can only have one mismatch */
         throw = temp - 1;
      }
   }

   return throw;
}


void gen_matrix(int **matrix, int **state, int n, int ns, int minthrow)
/* make entire matrix */
{
   int i, j, temp;

   for (i = 0; i < ns; i++)
      for (j = 0; j < ns; j++) {
         temp = matrix_element(state, i, j, n);
         if ((temp < minthrow) && (temp != 1))
            temp = 0;
         matrix[i][j] = temp;
      }
}


/*  gen_patterns  --  The following functions actually generate the  */
/*                    juggling patterns, using the throwing matrix   */
/*                    created above.                                 */

int gen_loops(int **matrix, int **state, int start, int from, int num,
      int *pattern, int pos, int *used, int n, int ns, int l, int numflag)
{
   int to, i, j, k;

   for (to = start; to < ns; to++)
      if ((matrix[from][to] != 0) && (used[to] == 0)) {
         if (to == start) {           /* we've formed a complete loop */
            if ((pos == l) || (l == -1)) {
               if (numflag != 2) {      /* should we print the pattern? */

                  pattern[pos] = matrix[from][start];

                  for (i = 0; i < n; i++) {     /* print startup sequence */
                     if (state[start][i] != (i + 1)) {
                        j = n - 1 - i + state[start][i];
                        if (j < 10)
                           printf("%d ", j);
                        else
                           printf("%c ", j - 10 + 'A');
                     } else
                        printf("  ");
                  }
                  printf(" ");

                  for (i = 0; i <= pos; i++) {       /* print the pattern */
                     j = pattern[i];
                     if (j < 10)
                        printf("%d ", j);
                     else
                        printf("%c ", j - 10 + 'A');
                  }
                  printf(" ");

                  for (i = j = 0, k = 1; j < n; j++) {   /* print ending */
                     while (++i < state[start][j]) {
                        if ((i - k) < 10)
                           printf("%d ", i - k);
                        else
                           printf("%c ", i - k - 10 + 'A');
                        k++;
                     }
                  }

                  printf("\n");
               }
               num++;                      /* increment running counter */
            }
         } else if ((pos < l) || (l == -1)) {
            pattern[pos] = matrix[from][to];
            used[to] = 1;
            num = gen_loops(matrix, state, start, to, num, pattern, pos + 1,
                  used, n, ns, l, numflag);
            used[to] = 0;
         }
      }

   return num;
}


int gen_patterns(int **matrix, int **state, int *pattern, int *used, int n,
      int ns, int l, int numflag)
{
   int i, num = 0;

   for (i = 0; i < ns; i++)
      num = gen_loops(matrix, state, i, i, num, pattern, 0, used, n, ns, l,
               numflag);

   return num;
}


int main(int argc, char **argv)
{
   int **matrix, **state, *pattern, *used;
   int n, h, ns, l = -1, numflag = 0, minthrow = 0;
   int np, i;

   if (argc < 3) {
      printf("J  --  A site swap generator    (Jack Boyce, Nov. 1990)\n");
      printf(
    "   Usage: %s <# objects> <max. height> [<word length>] [-n[o]] [-min x]\n",
         argv[0]);
      exit(0);
   }

   n = atoi(argv[1]);                    /* get the number of objects */
   if (n < 3) {
      printf("Must have at least 3 objects\n");
      exit(0);
   }
   h = atoi(argv[2]);                    /* get the max. throw height */
   if (h < n) {
      printf("Max. throw height must equal or exceed number of objects\n");
      exit(0);
   }
   ns = num_states(n, h);             /* find the number of states */

   for (i = 3; i < argc; i++) {
      if (!strcmp(argv[i], "-n"))
         numflag = 1;
      else if (!strcmp(argv[i], "-no"))
         numflag = 2;
      else if (!strcmp(argv[i], "-min")) {
         i++;
         minthrow = atoi(argv[i]);
      }
      else
         l = atoi(argv[i]) - 1;
   }

   if (minthrow > h) {
      printf("Min. throw height cannot be greater than maximum\n");
      exit(0);
   }

      /* Now allocate the memory space for the list of states,  */
      /* the throwing matrix, and other stuff.                  */

   if ((state = (int **)malloc((ns + 1) * sizeof(int *))) == 0)
      die();
   for (i = 0; i <= ns; i++)
      if ((state[i] = (int *)malloc(n * sizeof(int))) == 0)
         die();
   if ((matrix = (int **)malloc(ns * sizeof(int *))) == 0)
      die();
   for (i = 0; i < ns; i++)
      if ((matrix[i] = (int *)malloc(ns * sizeof(int))) == 0)
         die();
   if ((pattern = (int *)malloc(ns * sizeof(int))) == 0)
      die();
   if ((used = (int *)malloc(ns * sizeof(int))) == 0)
      die();

   (void)gen_states(state, 0, n - 1, n, h);    /* generate list of states */
   gen_matrix(matrix, state, n, ns, minthrow); /* generate throwing matrix */

   for (i = 0; i < ns; i++)
      used[i] = 0;

   np = gen_patterns(matrix, state, pattern, used, n, ns, l, numflag);

   if (numflag != 0)
      printf("%d patterns found\n", np);

   free(used);                       /* free all memory used by program */
   free(pattern);
   for (i = 0; i < ns; i++)
      free(matrix[i]);
   free(matrix);
   for (i = 0; i <= ns; i++)
      free(state[i]);
   free(state);

   return 0;
}


void die()
{
   printf("Insufficient memory\n");
   exit(0);
}
