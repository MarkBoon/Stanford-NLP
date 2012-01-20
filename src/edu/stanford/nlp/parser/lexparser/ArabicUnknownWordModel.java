package edu.stanford.nlp.parser.lexparser;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import edu.stanford.nlp.io.EncodingPrintWriter;
import edu.stanford.nlp.ling.LabeledWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Index;

/**
 * This is a basic unknown word model for Arabic.  It supports 4 different
 * types of feature modeling; see {@link #getSignature(String, int)}.
 *
 * <i>Implementation note: the contents of this class tend to overlap somewhat
 * with {@link EnglishUnknownWordModel} and were originally included in {@link BaseLexicon}.
 *
 * @author Dan Klein
 * @author Galen Andrew
 * @author Christopher Manning
 * @author Anna Rafferty
 */
public class ArabicUnknownWordModel extends BaseUnknownWordModel {

  private static final long serialVersionUID = 4825624957364628771L;

  private static final int MIN_UNKNOWN = 6;

  private static final int MAX_UNKNOWN = 10;
  String UNKNOWN_WORD = "UNK";  // if UNK were a word, counts would merge
  String BOUNDARY_TAG = ".$$."; // boundary tag -- assumed not a real tag

  protected boolean smartMutation = false;


  private static final boolean DOCUMENT_UNKNOWNS = false;

  protected int unknownSuffixSize = 0;
  protected int unknownPrefixSize = 0;

  public ArabicUnknownWordModel(Options op, Lexicon lex, Index<String> wordIndex, Index<String> tagIndex) {
    super(op, lex, wordIndex, tagIndex);
    unknownLevel = op.lexOptions.useUnknownWordSignatures;
    if (unknownLevel < MIN_UNKNOWN || unknownLevel > MAX_UNKNOWN) {
      if (unknownLevel < MIN_UNKNOWN) {
        unknownLevel = MIN_UNKNOWN;
      } else if (unknownLevel > MAX_UNKNOWN) {
        unknownLevel = MAX_UNKNOWN;
      }
      System.err.println("Invalid value for useUnknownWordSignatures");
    }
    this.smartMutation = op.lexOptions.smartMutation;
    this.unknownSuffixSize = op.lexOptions.unknownSuffixSize;
    this.unknownPrefixSize = op.lexOptions.unknownPrefixSize;
  }

  /**
   * Trains this lexicon on the Collection of trees.
   */
  @Override
  public void train(Collection<Tree> trees) {
    train(trees, 1.0, false);
  }


  /**
   * Trains this lexicon on the Collection of trees.
   *
   * @param trees The trees tro build a lexicon from
   * @param keepTagsAsLabels Whether tags should be represented as Labels or
   *     Strings in the lexicon.
   */
  public void train(Collection<Tree> trees, boolean keepTagsAsLabels) {
    train(trees, 1.0, keepTagsAsLabels);
  }

  public void train(Collection<Tree> trees, double weight) {
    train(trees, weight, false);
  }

  public void train(Collection<Tree> trees, double weight, boolean keepTagsAsLabels) {
    // Records the number of times word/tag pair was seen in training data.
    ClassicCounter<IntTaggedWord> seenCounter = new ClassicCounter<IntTaggedWord>();

    // scan data
    int tNum = 0;
    int tSize = trees.size();
    int indexToStartUnkCounting = (int) (tSize * trainOptions.fractionBeforeUnseenCounting);
    if (DOCUMENT_UNKNOWNS) {
      System.err.println("Collecting " + UNKNOWN_WORD + " from trees " +
          (indexToStartUnkCounting + 1) + " to " + tSize);
    }

    for (Tree tree : trees) {
      tNum++;
      List<IntTaggedWord> taggedWords = treeToEvents(tree, keepTagsAsLabels);
      for (int w = 0, sz = taggedWords.size(); w < sz; w++) {
        IntTaggedWord iTW = taggedWords.get(w);
        IntTaggedWord iT = new IntTaggedWord(nullWord, iTW.tag);
        IntTaggedWord iW = new IntTaggedWord(iTW.word, nullTag);
        seenCounter.incrementCount(iW, weight);
        IntTaggedWord i = NULL_ITW;

        if (tNum > indexToStartUnkCounting) {
          // start doing this once some way through trees; tNum is 1 based counting
          if (seenCounter.getCount(iW) < 2) {
            // it's an entirely unknown word
            int s = getSignatureIndex(iTW.word, w, wordIndex.get(iTW.word));
            if (DOCUMENT_UNKNOWNS) {
              String wStr = wordIndex.get(iTW.word);
              String tStr = tagIndex.get(iTW.tag);
              String sStr = wordIndex.get(s);
              EncodingPrintWriter.err.println("Unknown word/tag/sig:\t" +
                      wStr + '\t' + tStr + '\t' + sStr,
                      "UTF-8");
            }
            IntTaggedWord iTS = new IntTaggedWord(s, iTW.tag);
            IntTaggedWord iS = new IntTaggedWord(s, nullTag);
            unSeenCounter.incrementCount(iTS, weight);
            unSeenCounter.incrementCount(iT, weight);
            unSeenCounter.incrementCount(iS, weight);
            unSeenCounter.incrementCount(i, weight);
          } // else {
        }
      }
    }
    // make sure the unseen counter isn't empty!  If it is, put in
    // a uniform unseen over tags
    if (unSeenCounter.isEmpty()) {
      int numTags = tagIndex.size();
      for (int tt = 0; tt < numTags; tt++) {
        if ( ! BOUNDARY_TAG.equals(tagIndex.get(tt))) {
          IntTaggedWord iT = new IntTaggedWord(nullWord, tt);
          IntTaggedWord i = NULL_ITW;
          unSeenCounter.incrementCount(iT, weight);
          unSeenCounter.incrementCount(i, weight);
        }
      }
    }
    // index the possible tags for each word
    // numWords = wordIndex.size();
    // unknownWordIndex = wordIndex.indexOf(Lexicon.UNKNOWN_WORD, true);
    // initRulesWithWord();
  }

  protected List<IntTaggedWord> treeToEvents(Tree tree, boolean keepTagsAsLabels) {
    if (!keepTagsAsLabels) { return treeToEvents(tree); }
    List<LabeledWord> labeledWords = tree.labeledYield();
    return listOfLabeledWordsToEvents(labeledWords);
  }

  protected List<IntTaggedWord> treeToEvents(Tree tree) {
    List<TaggedWord> taggedWords = tree.taggedYield();
    return listToEvents(taggedWords);
  }

  protected List<IntTaggedWord> listToEvents(List<TaggedWord> taggedWords) {
    List<IntTaggedWord> itwList = new ArrayList<IntTaggedWord>();
    for (TaggedWord tw : taggedWords) {
      IntTaggedWord iTW = new IntTaggedWord(tw.word(), tw.tag(), wordIndex, tagIndex);
      itwList.add(iTW);
    }
    return itwList;
  }

  protected List<IntTaggedWord> listOfLabeledWordsToEvents(List<LabeledWord> taggedWords) {
    List<IntTaggedWord> itwList = new ArrayList<IntTaggedWord>();
    for (LabeledWord tw : taggedWords) {
      IntTaggedWord iTW = new IntTaggedWord(tw.word(), tw.tag().value(), wordIndex, tagIndex);
      itwList.add(iTW);
    }
    return itwList;
  }


  @Override
  public float score(IntTaggedWord iTW, int loc, double c_Tseen, double total, double smooth, String word) {
    double pb_W_T; // always set below

    //  unknown word model for P(T|S)

    int wordSig = getSignatureIndex(iTW.word, loc, word);
    IntTaggedWord temp = new IntTaggedWord(wordSig, iTW.tag);
    double c_TS = unSeenCounter.getCount(temp);
    temp = new IntTaggedWord(wordSig, nullTag);
    double c_S = unSeenCounter.getCount(temp);
    double c_U = unSeenCounter.getCount(NULL_ITW);
    temp = new IntTaggedWord(nullWord, iTW.tag);
    double c_T = unSeenCounter.getCount(temp);

    double p_T_U = c_T / c_U;

    if (unknownLevel == 0) {
      c_TS = 0;
      c_S = 0;
    }
    double pb_T_S = (c_TS + smooth * p_T_U) / (c_S + smooth);

    double p_T = (c_Tseen / total);
    double p_W = 1.0 / total;
    pb_W_T = Math.log(pb_T_S * p_W / p_T);

    return (float) pb_W_T;
  }


  /**
   * Returns the index of the signature of the word numbered wordIndex, where
   * the signature is the String representation of unknown word features.
   */
  @Override
  public int getSignatureIndex(int index, int sentencePosition, String word) {
    String uwSig = getSignature(word, sentencePosition);
    int sig = wordIndex.indexOf(uwSig, true);
    return sig;
  }

  /**
   *  6-9 were added for Arabic. 6 looks for the prefix Al- (and
   * knows that Buckwalter uses various symbols as letters), while 7 just looks
   * for numbers and last letter. 8 looks for Al-, looks for several useful
   * suffixes, and tracks the first letter of the word. (note that the first
   * letter seems a bit more informative than the last letter, overall.)
   * 9 tries to build on 8, but avoiding some of its perceived flaws: really it
   * was using the first AND last letter.
   *
   * @param word The word to make a signature for
   * @param loc Its position in the sentence (mainly so sentence-initial
   *          capitalized words can be treated differently)
   * @return A String that is its signature (equivalence class)
   */
  @Override
  public String getSignature(String word, int loc) {
    StringBuilder sb = new StringBuilder("UNK");
    switch (unknownLevel) {
    case 10://Anna's attempt at improving Chris' attempt, April 2008
    {
      boolean allDigitPlus = ArabicUnknownWordSignatures.allDigitPlus(word);
      int leng = word.length();
      if (allDigitPlus) {
        sb.append("-NUM");
      } else if (word.startsWith("Al") || word.startsWith("\u0627\u0644")) {
        sb.append("-Al");
      } else {
        // the first letters of a word seem more informative overall than the
        // last letters.
        // Alternatively we could add on the first two letters, if there's
        // enough data.
        if (unknownPrefixSize > 0) {
          int min = leng < unknownPrefixSize ? leng: unknownPrefixSize;
          sb.append('-').append(word.substring(0, min));
        }
      }
      if(word.length() == 1) {
        //add in the unicode type for the char
        sb.append(Character.getType(word.charAt(0)));
      }
      sb.append(ArabicUnknownWordSignatures.likelyAdjectivalSuffix(word));
      sb.append(ArabicUnknownWordSignatures.pastTenseVerbNumberSuffix(word));
      sb.append(ArabicUnknownWordSignatures.presentTenseVerbNumberSuffix(word));
      String ans = ArabicUnknownWordSignatures.abstractionNounSuffix(word);
      if (! "".equals(ans)) {
        sb.append(ans);
      } else {
        sb.append(ArabicUnknownWordSignatures.taaMarbuuTaSuffix(word));
      }
      if (unknownSuffixSize > 0 && ! allDigitPlus) {
        int min = leng < unknownSuffixSize ? leng: unknownSuffixSize;
        sb.append('-').append(word.substring(word.length() - min));
      }
      break;
    }
    case 9: // Chris' attempt at improving Roger's Arabic attempt, Nov 2006.
    {
      boolean allDigitPlus = ArabicUnknownWordSignatures.allDigitPlus(word);
      int leng = word.length();
      if (allDigitPlus) {
        sb.append("-NUM");
      } else if (word.startsWith("Al") || word.startsWith("\u0627\u0644")) {
        sb.append("-Al");
      } else {
        // the first letters of a word seem more informative overall than the
        // last letters.
        // Alternatively we could add on the first two letters, if there's
        // enough data.
        if (unknownPrefixSize > 0) {
          int min = leng < unknownPrefixSize ? leng: unknownPrefixSize;
          sb.append('-').append(word.substring(0, min));
        }
      }

      sb.append(ArabicUnknownWordSignatures.likelyAdjectivalSuffix(word));
      sb.append(ArabicUnknownWordSignatures.pastTenseVerbNumberSuffix(word));
      sb.append(ArabicUnknownWordSignatures.presentTenseVerbNumberSuffix(word));
      String ans = ArabicUnknownWordSignatures.abstractionNounSuffix(word);
      if (! "".equals(ans)) {
        sb.append(ans);
      } else {
        sb.append(ArabicUnknownWordSignatures.taaMarbuuTaSuffix(word));
      }
      if (unknownSuffixSize > 0 && ! allDigitPlus) {
        int min = leng < unknownSuffixSize ? leng: unknownSuffixSize;
        sb.append('-').append(word.substring(word.length() - min));
      }
      break;
    }

    case 8: // Roger's attempt at an Arabic UWM, May 2006.
    {
      if (word.startsWith("Al")) {
        sb.append("-Al");
      }
      boolean allDigitPlus = ArabicUnknownWordSignatures.allDigitPlus(word);
      if (allDigitPlus) {
        sb.append("-NUM");
      } else {
        // the first letters of a word seem more informative overall than the
        // last letters.
        // Alternatively we could add on the first two letters, if there's
        // enough data.
        sb.append('-').append(word.charAt(0));
      }
      sb.append(ArabicUnknownWordSignatures.likelyAdjectivalSuffix(word));
      sb.append(ArabicUnknownWordSignatures.pastTenseVerbNumberSuffix(word));
      sb.append(ArabicUnknownWordSignatures.presentTenseVerbNumberSuffix(word));
      sb.append(ArabicUnknownWordSignatures.taaMarbuuTaSuffix(word));
      sb.append(ArabicUnknownWordSignatures.abstractionNounSuffix(word));
      break;
    }

    case 7: {
      // For Arabic with Al's separated off (cdm, May 2006)
      // { -NUM, -lastChar }
      boolean allDigitPlus = ArabicUnknownWordSignatures.allDigitPlus(word);
      if (allDigitPlus) {
        sb.append("-NUM");
      } else {
        sb.append(word.charAt(word.length() - 1));
      }
      break;
    }

    case 6: {
      // For Arabic (cdm, May 2006), with Al- as part of word
      // { -Al, 0 } +
      // { -NUM, -last char(s) }
      if (word.startsWith("Al")) {
        sb.append("-Al");
      }
      boolean allDigitPlus = ArabicUnknownWordSignatures.allDigitPlus(word);
      if (allDigitPlus) {
        sb.append("-NUM");
      } else {
        sb.append(word.charAt(word.length() - 1));
      }
      break;
    }
    default:
      // 0 = do nothing so it just stays as "UNK"
    } // end switch (unknownLevel)
    // System.err.println("Summarized " + word + " to " + sb.toString());
    return sb.toString();
  } // end getSignature()


  @Override
  public void setUnknownLevel(int unknownLevel) {
    this.unknownLevel = unknownLevel;
  }

  @Override
  public int getUnknownLevel() {
    return unknownLevel;
  }

}
