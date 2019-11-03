// HTTF2018 予選　山形足し算
// 初心者編やった：https://qiita.com/tsukammo/items/7041a00e429f9f5ac4ae

import java.io.*;
import java.util.*;
import java.lang.*;

class Main {

    Scanner sc = new Scanner(System.in);

    public static void main(String[] args) throws IOException {
        new Main().solve();
    }

    final static int N = 100, ROW = 100, COL =  100, PNUM = 1000;

    void solve() {
        input();
        init();
        simulate();
        output();
        // writeLog();
    }

    int[][] map = new int[ROW][COL];

    void input() {
        for (int i = 0; i < ROW; i++){
            for (int j = 0; j < COL; j++) {
                map[j][i] = sc.nextInt();
            }
        }
    }

    Random rnd = new Random(20191101);
    int[][] ans = new int[PNUM][3];

    void init() {
        for (int i = 0; i < PNUM; i++) {
            int x = rnd.nextInt(N);
            int y = rnd.nextInt(N);
            int h = rnd.nextInt(N) + 1;
            ans[i][0] = x;
            ans[i][1] = y;
            ans[i][2] = h;
        }
    }

    // 実行制限: 6000 ms
    static final long TIME_LIMIT = 5500;

    void simulate() {
        long st = System.currentTimeMillis();
        long et = st + TIME_LIMIT;

        int bestScore = eval(ans);
        int[][] bestOutput = new int[1000][3];
        for (int i = 0; i < bestOutput.length; i++) {
            bestOutput[i] = Arrays.copyOf(ans[i], ans[i].length);
        }
        while (System.currentTimeMillis() < et) {
            int[][] tmpOutput = new int[1000][3];
            for (int i = 0; i < PNUM; i++) {
                int x = rnd.nextInt(N);
                int y = rnd.nextInt(N);
                int h = rnd.nextInt(N) + 1;
                tmpOutput[i][0] = x;
                tmpOutput[i][1] = y;
                tmpOutput[i][2] = h;
            }
            int tmpScore = eval(tmpOutput);
            if (bestScore > tmpScore) {
                bestScore = tmpScore;
                for (int i = 0; i < bestOutput.length; i++) {
                    bestOutput[i] = Arrays.copyOf(tmpOutput[i], tmpOutput[i].length);
                }
            }
        }

        for (int i = 0; i < bestOutput.length; i++) {
            ans[i] = Arrays.copyOf(bestOutput[i], bestOutput[i].length);
        }
    }

    // 山の外周に沿ったぐるぐる
    final static int dxs[] = new int[]{ 1, -1, -1, 1 };
    final static int dys[] = new int[]{ 1, 1, -1, -1 };

    int eval(int[][] output) {
        // 返り血が低いほど良い
        int ret = 0;
        int[][] ansMap = new int[ROW][COL];

        for (int i = 0; i < output.length; i++) {
            int x = output[i][0];
            int y = output[i][1];
            int h = output[i][2];
            ansMap[x][y] += h;
            for (int plus = 1; plus < h; plus++) {
                int d = h - plus;
                x = output[i][0];
                y = output[i][1] - d;
                for (int j = 0; j < dxs.length; j++) {
                    for (int k = 0; k < d; k++) {
                        x += dxs[j];
                        y += dys[j];
                        if (isOutOfMap(x, y)) {
                            continue;
                        }
                        ansMap[x][y] += plus;
                    }
                }
            }
        }

        for (int i = 0; i < ROW; i++) {
            for (int j = 0; j < COL; j++) {
                ret += Math.abs(map[i][j] - ansMap[i][j]);
            }
        }

        return ret;
    }

    boolean isOutOfMap(int x, int y) {
        return !((-1 < x && x < ROW) && (-1 < y && y < COL));
    }

    void output() {
        System.out.println(ans.length);
        for (int i = 0; i < ans.length; i++) {
            System.out.println(String.format("%d %d %d", ans[i][0], ans[i][1], ans[i][2]));
        }
    }


}