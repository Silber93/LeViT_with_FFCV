import argparse
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt

def get_args_parser():
    parser = argparse.ArgumentParser(
        'analysis script', add_help=False)
    parser.add_argument('--model-name')
    parser.add_argument('--levit-model-dir')
    parser.add_argument('--ffcv-model-dir')
    return parser


def get_time_acc(df: pd.DataFrame, version):
    df['train_dur'] = pd.to_timedelta(df['train_dur'])
    val_time = pd.to_timedelta(df['val_eval_dur']).mean()
    sums = [(df['train_dur'].iloc[range(i+1)].sum() + val_time).total_seconds() / 60 for i in range(df.shape[0])]
    m_acc, argmax = 0, 0
    accs = df['val_acc1'].values
    # for i, acc in enumerate(accs):
    #     m_acc, argmax = (acc, i) if acc > m_acc else (m_acc, argmax)
    # sums = sums[:argmax+1]
    # accs = df['val_acc1'].values[:argmax+1]
    plt.scatter(sums, accs)
    plt.plot(sums, accs, label=version)
    return df[['train_dur', 'val_acc1']].dropna()


def get_deltatimes(durations):
    print(durations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            'analysis script', parents=[get_args_parser()])
    args = parser.parse_args()
    levit_df = pd.read_csv(f'{args.levit_model_dir}/logged_data.csv')
    ffcv_df = pd.read_csv(f'{args.ffcv_model_dir}/logged_data.csv')
    get_time_acc(levit_df.dropna(), 'LeViT')
    get_time_acc(ffcv_df.dropna(), 'FFCV')
    plt.title(f'training time - LeViT_{args.model_name}')
    plt.xlabel('time (minutes)')
    plt.ylabel('accuracy (%)')
    plt.grid()
    plt.legend()
    plt.savefig(f'{args.model_name}_comparison.png')
