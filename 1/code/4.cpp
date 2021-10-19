#include <iostream>
#include <cmath>

using namespace std;

double intres(int x, double q)
{
    return 4 * M_PI * (x - q / 2 * log((x + q) / x - q));
}

double sumres(int x, double q)
{
    double res = 0;
    for (int i = -x; i <= x; i++)
    {
        // cout << i << endl;
        for (int j = -x; j <= x; j++)
        {
            for (int k = -x; k <= x; k++)
            {
                int n2 = i * i + j * j + k * k;
                if (n2 < x * x)
                {
                    res += 1 / (n2 - q * q);
                }
            }
        }
    }
    return res;
}

double sumreseq(int x, double q)
{
    double res = 0;
    for (int i = -x; i <= x; i++)
    {
        // cout << i << endl;
        for (int j = -x; j <= x; j++)
        {
            for (int k = -x; k <= x; k++)
            {
                int n2 = i * i + j * j + k * k;
                if (n2 == x * x)
                {
                    res += 1 / (n2 - q * q);
                }
            }
        }
    }
    return res;
}

double deltares(int Rl, int Rh, double q)
{
    double res = 0;
    for (int i = -Rh; i <= Rh; i++)
    {
        // cout << i << endl;
        for (int j = -Rh; j <= Rh; j++)
        {
            for (int k = -Rh; k <= Rh; k++)
            {
                int n2 = i * i + j * j + k * k;
                if (n2 < Rh * Rh && n2 >= Rl * Rl)
                {
                    res += 1 / (n2 - q * q);
                }
            }
        }
    }
    return res;
}

int main()
{
    // 从100到2000，输出表面项的差距
    for (int R = 100; R <= 2000; R += 100)
    {
        double f = sumreseq(R, sqrt(0.5));
        cout << f << endl;
    }
}