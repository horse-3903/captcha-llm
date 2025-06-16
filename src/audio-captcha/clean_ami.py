import os
import logging
from pydub import AudioSegment
from lxml import etree

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("segment_extraction.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# Paths
root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
audio_dir = os.path.join(root_path, "data/audio-captcha/ami-corpus")
transcript_dir = os.path.join(root_path, "data/audio-captcha/ami-transcriptions/words")
output_dir = os.path.join(root_path, "data/audio-captcha/ami-data-segments")
os.makedirs(output_dir, exist_ok=True)

# Parse transcript XML file with local-name() XPath (namespace agnostic)
def parse_transcript(xml_path):
    try:
        tree = etree.parse(xml_path)  # type: ignore
        root = tree.getroot()

        # Use local-name() to ignore namespaces
        word_elements = root.xpath('//*[local-name()="w"]')

        segments = []
        for w in word_elements:
            try:
                start = float(w.get('starttime'))
                end = float(w.get('endtime'))
                text = w.text or ""
                segments.append((start, end, text.strip()))
            except Exception as e:
                logging.warning(f"Skipping word in {os.path.basename(xml_path)}: {e}")

        logging.info(f"Parsed {len(segments)} words from {os.path.basename(xml_path)}")
        return segments

    except Exception as e:
        logging.error(f"Failed to parse XML {xml_path}: {e}")
        return []

# Extract transcript for a segment
def get_transcript(words, start_time, end_time):
    # Include words that at least partially overlap with the segment
    return " ".join([w[2] for w in words if w[0] < end_time and w[1] > start_time])

# Process all full-meeting WAV files
for file_name in os.listdir(audio_dir):
    if not file_name.endswith(".wav"):
        continue

    base_meeting = file_name.removesuffix(".Mix-Headset.wav")
    audio_path = os.path.join(audio_dir, file_name)

    # Find all speaker XMLs like EN2001a.A.words.xml, EN2001a.B.words.xml, etc.
    speaker_xmls = [f for f in os.listdir(transcript_dir) if f.startswith(base_meeting + ".") and f.endswith(".words.xml")]

    if not speaker_xmls:
        logging.warning(f"No transcript XMLs found for {base_meeting}")
        continue

    # Merge all speaker transcripts
    all_words = []
    for xml_file in speaker_xmls:
        full_path = os.path.join(transcript_dir, xml_file)
        words = parse_transcript(full_path)
        all_words.extend(words)

    # Sort by start time
    all_words.sort(key=lambda x: x[0])

    try:
        audio = AudioSegment.from_wav(audio_path)
        duration_sec = len(audio) / 1000

        logging.info(f"Processing {base_meeting}: duration = {duration_sec:.2f}s, {len(all_words)} words")

        for i in range(0, int(duration_sec), 10):
            segment_start = i
            segment_end = min(i + 10, duration_sec)

            # Create folder for this segment
            segment_folder = os.path.join(output_dir, f"{base_meeting}_{i:04d}")
            os.makedirs(segment_folder, exist_ok=True)

            segment = audio[segment_start * 1000 : segment_end * 1000]
            segment_file = os.path.join(segment_folder, f"{base_meeting}_{i:04d}.wav")
            segment.export(segment_file, format="wav")

            segment_text = get_transcript(all_words, segment_start, segment_end)
            text_file = os.path.join(segment_folder, f"{base_meeting}_{i:04d}.txt")
            with open(text_file, "w", encoding="utf-8") as f:
                f.write(segment_text)

            logging.debug(f"Exported {segment_file} with transcript: {segment_text[:50]}...")

    except Exception as e:
        logging.error(f"Error processing {base_meeting}: {e}")

logging.info("Done extracting segments and transcripts.")
