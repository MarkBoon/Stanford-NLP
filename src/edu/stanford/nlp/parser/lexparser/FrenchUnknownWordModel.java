package edu.stanford.nlp.parser.lexparser;

import edu.stanford.nlp.international.french.FrenchUnknownWordSignatures;
import edu.stanford.nlp.ling.LabeledWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.stats.ClassicCounter;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.util.Index;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;


public class FrenchUnknownWordModel extends BaseUnknownWordModel {

  private static final long serialVersionUID = -776564693549194424L;

  protected boolean smartMutation = false;

  protected int unknownSuffixSize = 0;
  protected int unknownPrefixSize = 0;

  private static final String BOUNDARY_TAG = ".$$."; // boundary tag -- assumed not a real tag

  public FrenchUnknownWordModel(Options op, Lexicon lex, Index<String> wordIndex, Index<String> tagIndex) {
    super(op, lex, wordIndex, tagIndex);
    unknownLevel = op.lexOptions.useUnknownWordSignatures;
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


  private void train(Collection<Tree> trees, double weight, boolean keepTagsAsLabels) {
    ClassicCounter<IntTaggedWord> seenCounter = new ClassicCounter<IntTaggedWord>();

    int tNum = 0;
    int tSize = trees.size();
    int indexToStartUnkCounting = (int) (tSize * trainOptions.fractionBeforeUnseenCounting);

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
            IntTaggedWord iTS = new IntTaggedWord(s, iTW.tag);
            IntTaggedWord iS = new IntTaggedWord(s, nullTag);
            unSeenCounter.incrementCount(iTS, weight);
            unSeenCounter.incrementCount(iT, weight);
            unSeenCounter.incrementCount(iS, weight);
            unSeenCounter.incrementCount(i, weight);
          }
        }
      }
    }
    // make sure the unseen counter isn't empty!  If it is, put in
    // a uniform unseen over tags
    if (unSeenCounter.isEmpty()) {
      System.err.printf("%s: WARNING: Unseen word counter is empty!%n",this.getClass().getName());
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
   * TODO Can add various signatures, setting the signature via Options.
   *
   * @param word The word to make a signature for
   * @param loc Its position in the sentence (mainly so sentence-initial
   *          capitalized words can be treated differently)
   * @return A String that is its signature (equivalence class)
   */
  @Override
  public String getSignature(String word, int loc) {
    final String BASE_LABEL = "UNK";
    StringBuilder sb = new StringBuilder(BASE_LABEL);
    switch (unknownLevel) {
      case 1: //Marie's initial attempt
        sb.append(FrenchUnknownWordSignatures.nounSuffix(word));
        if(sb.toString().equals(BASE_LABEL)) {
          sb.append(FrenchUnknownWordSignatures.adjSuffix(word));
          if(sb.toString().equals(BASE_LABEL)) {
            sb.append(FrenchUnknownWordSignatures.verbSuffix(word));
            if(sb.toString().equals(BASE_LABEL)) {
              sb.append(FrenchUnknownWordSignatures.advSuffix(word));
            }
          }
        }

        sb.append(FrenchUnknownWordSignatures.possiblePlural(word));

        String hasDigit = FrenchUnknownWordSignatures.hasDigit(word);
        String isDigit = FrenchUnknownWordSignatures.isDigit(word);

        if( ! hasDigit.equals("")) {
          if(isDigit.equals("")) {
            sb.append(hasDigit);
          } else {
            sb.append(isDigit);
          }
        }

//        if(FrenchUnknownWordSignatures.isPunc(word).equals(""))
          sb.append(FrenchUnknownWordSignatures.hasPunc(word));
//        else
//          sb.append(FrenchUnknownWordSignatures.isPunc(word));

        sb.append(FrenchUnknownWordSignatures.isAllCaps(word));

        if(loc > 0) {
          if(FrenchUnknownWordSignatures.isAllCaps(word).equals(""))
            sb.append(FrenchUnknownWordSignatures.isCapitalized(word));
        }

        //Backoff to suffix if we haven't matched anything else
        if(unknownSuffixSize > 0 && sb.toString().equals(BASE_LABEL)) {
          int min = word.length() < unknownSuffixSize ? word.length(): unknownSuffixSize;
          sb.append('-').append(word.substring(word.length() - min));
        }

        break;

      default:
        System.err.printf("%s: Invalid unknown word signature! (%d)%n", this.getClass().getName(),unknownLevel);
    }

    return sb.toString();
  }
}
