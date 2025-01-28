def check_docstring_content(function_docstring):
    """
    Check the content of a function's docstring for specific sections.

    Identifies if the docstring likely contains Args, Returns, Raises, or Examples.
    Only meaningful if docstring has multiple lines.

    Args:
        function_docstring (str): The docstring of the function to check.

    Returns:
        dict: A dictionary indicating the presence of the following sections:
              - 'HasDocstring' (bool): True if a docstring is present at all.
              - 'Args' (bool): True if 'Args:', 'Arguments:', 'Params:', or 'Parameters:' is present.
              - 'Returns' (bool): True if 'Returns:' or 'Yields:' is present.
              - 'Raises' (bool): True if 'Raises:' or 'Exceptions:' is present.
              - 'Examples' (bool): True if 'Examples:', 'Example:', or 'Usage:' is present.
              If the docstring is None, all keys will return False.
    """
    if not function_docstring:
        return {
            'HasDocstring': False,
            'HasMultipleLines': False,
            'Args': False,
            'Returns': False,
            'Raises': False,
            'Examples': False
        }

    has_multiple_line = len(function_docstring.strip().split('\n')) <= 1

    args_keywords = ['Args:', 'Arguments:', 'Params:', 'Parameters:']
    returns_keywords = ['Returns:', 'Yields:']
    raises_keywords = ['Raises:', 'Exceptions:']
    examples_keywords = ['Examples:', 'Example:', 'Usage:']

    results = {
        'HasDocstring': True,
        'HasMultipleLines': has_multiple_line,
        'Args': any(keyword in function_docstring for keyword in args_keywords),
        'Returns': any(keyword in function_docstring for keyword in returns_keywords),
        'Raises': any(keyword in function_docstring for keyword in raises_keywords),
        'Examples': any(keyword in function_docstring for keyword in examples_keywords),
    }
    return results


def analyze_file(filename):
    """
    Analyze a Python file for docstring quality.

    - Extracts all functions.
    - For each function, runs analysis and scores docstring quality.
    - Prints out results.

    Args:
        filename (str): Path to the Python file to analyze.

    Returns:
        list: A list of dicts containing the analysis for each function.
    """
    print(f'Analyzing file: {filename}')

    functions = extract_functions_from_file(filename)
    if not functions:
        print('No functions found in the file.')
        return []

    results = []

    for func in functions:
        function_name, function_source, function_doc, num_args, code_length = func
        print(f'\n--- Analyzing function: {function_name} ---')
        analysis_results = analyze_function(function_name, function_source, function_doc, num_args, code_length)
        start_line = func[3]
        results.append({
            'Name': function_name,
            'Docstring': function_doc,
            'Source': function_source,
            'IsComplex': analysis_results['is_complex'],
            'Score': analysis_results['score'],
            'StartLine': start_line,
            'CodeLength': code_length
        })

    return results


def _variable_on_cpu(name, shape, initializer, use_fp16=False):
    """
    Helper to create a Variable stored on CPU memory.

    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable

    Returns:
        Variable Tensor.
    """
    with tf.device('/cpu:0'):
        dtype = tf.float16 if use_fp16 else tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def read_fastqs(fastqlist, maxreads=None, reads_per_dot=100):
    """
    Read fastq files and calculate length and mean q score for each.

    Args:
        fastqlist (array of str): list of fastq files
        maxreads (int, optional): max number of files to read, default `None`
            is no maximum.
        reads_per_dot (int, optional): Print a dot for every `reads_per_dot`
            files.

    Returns:
        tuple of :class:`ndarray` and :class:`ndarray` and :class:`ndarray`:
            First element contains read_ids, the second is the corresponding
            mean q-score (NaN if no data was found), and the length of each
            base call.
    """
    read_id_list = []
    mean_qscore_list = []
    length_list = []
    print('Printing one dot for every {} reads.'.format(reads_per_dot))
    for fastqfile in fastqlist:
        for record in SeqIO.parse(fastqfile, 'fastq'):
            read_id_list.append(record.id)
            scores = np.array(record.letter_annotations['phred_quality'])
            length_list.append(len(scores))
            if len(scores) > 0:
                mean_qscore_list.append(fastq_file_qscore(scores))
            else:
                mean_qscore_list.append(None)
            if (len(read_id_list) + 1) % reads_per_dot == 0:
                sys.stdout.write('.')
            if maxreads is not None:
                if len(read_id_list) >= maxreads:
                    break
        if maxreads is not None:
            if len(read_id_list) >= maxreads:
                break
    print('')
    return (np.array(read_id_list), np.array(mean_qscore_list), np.array(length_list))


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`.

    Features supported:
        - MFCC (mfcc)
        - Chroma (chroma)
        - MEL Spectrogram Frequency (mel)
        - Contrast (contrast)
        - Tonnetz (tonnetz)

    e.g:
    `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get('mfcc')
    chroma = kwargs.get('chroma')
    mel = kwargs.get('mel')
    contrast = kwargs.get('contrast')
    tonnetz = kwargs.get('tonnetz')
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            pass
    except RuntimeError:
        basename = os.path.basename(file_name)
        dirname = os.path.dirname(file_name)
        (name, ext) = os.path.splitext(basename)
        new_basename = f'{name}_c.wav'
        new_filename = os.path.join(dirname, new_basename)
        v = convert_audio(file_name, new_filename)
        if v:
            raise NotImplementedError('Converting the audio files failed, make sure `ffmpeg` is installed in your machine and added to PATH.')
    else:
        new_filename = file_name
    with soundfile.SoundFile(new_filename) as sound_file:
        X = sound_file.read(dtype='float32')
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T, axis=0)
            result = np.hstack((result, tonnetz))
    return result


def run_pylint(filename):
    """Execute pylint to check for missing docstrings."""
    result = subprocess.run(
        [
            'pylint',
            '--disable=all',
            '--enable=missing-docstring',
            filename
        ],
        capture_output=True,
        text=True
    )
    return result.stdout


def run_pydocstyle(filename):
    """Execute pydocstyle to check for PEP 257 compliance."""
    result = subprocess.run(
        [
            'pydocstyle',
            '--add-ignore=D100',
            filename,
        ],
        capture_output=True,
        text=True,
        check=False
    )
    return result.stdout.strip().split('\n')


def update_progress(self, current, total):
    """Update the progress bar."""
    progress_percent = max(1, int((current / total) * 100))
    self.progress_bar['value'] = progress_percent
    self.progress_label.config(text=f'Progress: {progress_percent}%')
    self.update_idletasks()


def diff_lists(a, b):
    """Return a list of ScriptResultDiffs from two sorted lists of ScriptResults."""
    diffs = []
    i = 0
    j = 0
    while i < len(a) and j < len(b):
        if a[i].id < b[j].id:
            diffs.append(ScriptResultDiff(a[i], None))
            i += 1
        elif a[i].id > b[j].id:
            diffs.append(ScriptResultDiff(None, b[j]))
            j += 1
        else:
            if a[i].output != b[j].output or verbose:
                diffs.append(ScriptResultDiff(a[i], b[j]))
            i += 1
            j += 1
    while i < len(a):
        diffs.append(ScriptResultDiff(a[i], None))
        i += 1
    while j < len(b):
        diffs.append(ScriptResultDiff(None, b[j]))
        j += 1
    return diffs


def display_table(table):
    """
    Display values in a table format.

    Args:
        table: an iterable of rows, and each row is an iterable of values.
    """
    html = ''
    for row in table:
        row_html = ''
        for col in row:
            row_html += '<td>{:40}</td>'.format(str(col))
        html += '<tr>' + row_html + '</tr>'
    html = '<table>' + html + '</table>'
    IPython.display.display(IPython.display.HTML(html))
