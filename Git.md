# Git メモ

## git status等で日本語ファイル名が”\xxx\xxx”と数字の羅列になる

git statusコマンド等で日本語ファイル名が化ける場合はGitBash上から以下のコマンドを実行します。

```
$ git config --global core.quotepath false
```

これで日本語ファイル名も化けずに確認できる様になります。

git diff などは普通に日本語だったのでびびった（2019/11/17）



## cmdからファイルを所定のプログラムで開きたい

`start file_name` で開ける

macだと`open`みたい



## localに新しいブランチを作って，GitHubから特定のブランチをpullする

`git pull origin pullしたいリモートブランチ名:ローカルブランチ名`

例えば，`git pull origin enhanceA:enhanceA`



## いらないブランチを消す

`git branch -d branch_name`




### git log

- `--oneline`：一行で
- `--graph`：ソースツリーみたい



## 逆マージ

```
git checkout enhance
git merge develop
```

**コミットが済んでいないファイルが残っている状態で git merge しない**

コミットが済んでいない状態で git merge を行うと、ブランチの変更の取り込み漏れの原因になります。全てコミットが済んでいる状態で git merge するようにしましょう。



## マージを元に戻す方法

ここでは、マージを元に戻す方法をご紹介します。

上記のように、git merge により衝突が発生しても、修正方法はありますが、マージ自体を元に戻したい場合もあります。その場合、マージ後にコミットしたか、していないかによりコマンドが異なります。

マージした後、未だコミットしていない場合

```
git reset --hard HEAD
```

マージした後、コミットも行っている場合

```
git reset --hard ORIG_HEAD
```



## コミットメッセージを書き換えたい！！

#### 直前のコミットの場合

```
git commit -m "〇を変更う"
git commit --amend -m "〇を変更"
```

[Gitのコミットメッセージを後から変更する方法をわかりやすく書いてみた](https://www.granfairs.com/blog/staff/git-commit-fix)

二つ以上の前でも変更できるようだ



## HEADに戻りたい

```
git reset --hard HEAD
```

※もちろんUntrackedは対象外。Untrackedを消したい場合は、コメントで教えてもらった「git clean -f」を使う。もしくは、「git add .」して一旦管理下に入れて上記を行う。



### ファイル単位で戻したい

`reset --hard`はファイル単位ではできないので…

#### 対象がadd済みファイル(インデックス&ワーキングツリーに存在)

まずインデックスを戻して、次にワーキングツリーを戻す。

```
git reset HEAD <file>
git checkout <file>
```

#### 対象が未addファイル(ワーキングツリーのみに存在)

ワーキングツリーを戻す。

```
git checkout <file>
```

## 直前のコミットを消したい

```
git reset --hard HEAD^
```

## コミットを消すが履歴を残す場合（打消しのコミットをする）

```
git revert コミットのハッシュ地
```



## ブランチを消したい

### リモートブランチ

```
git push --delete origin branch_name
```

### ゾンビ化したとき

リモートリポジトリを直接消すと参照は残って消せなくなる

```
git push --delete origin branch_name
error: unable to delete 'branch_name': remote ref does not exist
error: failed to push some refs to 'https://github.com/dannyso16/hogehoge.git'
```

pruneで刈り取る

```
git remote prune origin
```




## Git push の取り消し

### 過ちを隠す（commitを取り消して，リモートから消す）

#### ローカルも書き換えるとき

```c
// 直前のコミットなら… ひとつ前に戻す
git reset --hard HEAD^

// 強制的にpush
git push -f origin (remote branch 名)
```

#### ローカルはそのままにしたい（未確認）

```c
// リモートの参照を強制的に戻す
git push -f origin {sha-1 like 8dj2dga...}:{remote branch name}
```



### 過ちを認める（間違いコミットを打ち消すrevertコミットを作って追加でpush）

安全だけどログが汚くなる

```ba
// 直前のコミットを打ち消すコミット
git revert HEAD
// 最新に加えて，さらに過去３つのコミットを打ち消す
git revert HEAD~3

// 新しいコミットなのでconflictしないし，-fで強制しなくていい
git push origin {branch name}
```

### 特定のコミットを確認したい

```bash
git show {shaハッシュ値}
```



## 絶望のCONFLICT

```bash
$ git pull
Auto-merging src/Git/Gitメモ.md
CONFLICT (content): Merge conflict in src/Git/Gitメモ.md
Automatic merge failed; fix conflicts and then commit the result.
```

うわああああ

```bash
$ git status
Unmerged paths:
  (use "git add <file>..." to mark resolution)
        both modified:   src/Git/Gitメモ.md
```

`both modified`らしい

言われた通り`git add `する

ファイルを確認すると以下のように変更されてる

```bash
<<<<<<< HEAD
# 作業ブランチでの変更内容
・・・
=======
# develop(マージしたブランチ)での変更内容
・・・
>>>>>>> develop
```

先にコンフリクトを自分で解消しておいてから、`add　→　commit`してもいいけど、CONFLICTの差分を残すためにも先にコミットしておいて、あとで修正コミットするのがいいのかな。

# Git の中身

## コミットハッシュ値

[Gitのコミットハッシュ値は何を元にどうやって生成されているのか](https://tech.mercari.com/entry/2016/02/08/173000)

[Gitのつくりかた](https://tech.mercari.com/entry/2015/09/14/175300)

```bash
git log -1
commit 757cd618f38d574238bae4768ff1a1aedfafdb7a
Author: DQNEO <dqneo@example.com>
Date:   Thu Feb 4 21:18:28 2016 +0900
```



コミットオブジェクトの中身を見る

```
$ git cat-file -p 757cd618f38d574238bae4768ff1a1aedfafdb7a
tree 05520e3bd0354e823cacf96b244987f235b3c240
parent 2476c4c7bcbf98e444b6851d67036077334502d2
author DQNEO <dqneo@example.com> 1454588308 +0900
committer DQNEO <dqneo@example.com> 1454588308 +0900

second commit
```

なんと実態は数行のテキストデータ

### ハッシュの計算式

```
hash = sha1("commit<半角スペース><コミットオブジェクトのバイト数>\0<コミットオブジェクトの中身>")
```



### shaの短縮など

[公式]([https://git-scm.com/book/ja/v2/Git-%E3%81%AE%E3%81%95%E3%81%BE%E3%81%96%E3%81%BE%E3%81%AA%E3%83%84%E3%83%BC%E3%83%AB-%E3%83%AA%E3%83%93%E3%82%B8%E3%83%A7%E3%83%B3%E3%81%AE%E9%81%B8%E6%8A%9E#r_commit_ranges](https://git-scm.com/book/ja/v2/Git-のさまざまなツール-リビジョンの選択#r_commit_ranges))



### ニュースなど

[hashを意図的に衝突させる攻撃に対するgoogle の生命](https://security.googleblog.com/2017/02/announcing-first-sha1-collision.html)



# python + git

subprocessを使う→コマンドを使える！→当然Gitもできる

```python
import subprocess
 
cmd = "git status"
```

[subprocess doc]( https://docs.python.org/ja/3/library/subprocess.html )

## call

```python
run_cmd = subprocess.call(cmd.split())
print(run_cmd) # return 0
```

実行の際はスペース区切りで渡す

もし`python child.py`のように実行すると，callはchildの実行が終わるまで待機する

返り血は終了時のコード０とか



## check_out

```python
run_cmd = subprocess.check_out(cmd.split())
print(run_cmd) # byte 文字列
```

byte文字列→b'hello\n world\r\n' みたいなん

コマンドラインで表示される文字列が返ってくる

strにするには `byte_str.decode("utf-8")`



## Popen

もし`python child.py`のように実行すると，Popenオブジェクトがかえって来る．これは新しいプロセス(child)を生成している．



# Linux commands



## ファイル操作

### 空のファイルを作る

`touch XXX.txt`

### ファイルのコピー

`cp コピーするファイル名 コピー先のファイル名`

例えば，`cp test.txt test_copy.txt`

### 名前の変更

`mv 変更したいファイル 変更後の名前`

例えば，`mv test.txt renamed.txt`

### 閲覧

`cat ファイル名`

### ファイルの削除

`rm ファイル名`

### エクスプローラで開く

`start .` ：今のディレクトリを開く



## システム

### シャットダウン関係(Linux)

`shutdown 時間`

例えば，`shutdown now`：電源は切らない

`shutdown -h +1` ：1分後にシステム停止し，電源を切る

`shutdown -r now`：再起動

`shutdown -c`：予定されたシャットダウンをキャンセル

### ※Windwosの場合

`shutdown -[option]`で時間はなくてもいい

| オプション            | 説明                                                         |
| --------------------- | ------------------------------------------------------------ |
| -i                    | GUIインターフェイスを表示する                                |
| -l                    | ログオフを実行する（-mと併用不可）                           |
| -s                    | シャットダウンする                                           |
| -r                    | コンピュータを再起動する                                     |
| -a                    | シャットダウンを中止する                                     |
| -m *\\Compurter Name* | リモート作業時のコンピュータ名を指定する                     |
| -t *sec*              | シャットダウンまでの時間を指定する                           |
| -c *"Comment"*        | シャットダウン時のメッセージ（コメント）を指定する（最大127文字） |
| -f                    | アプリケーションを強制終了する                               |
| -d (u\|p):xx(:yy)     | シャットダウンの理由コードを指定する（下記参照）             |

[すべてのオプション](https://www.k-tanaka.net/cmd/shutdown.php)