enum CharacteristicName { Dominance, Openness, Conscientiousness }

class Characteristic {
  CharacteristicName characteristicName;
  double value;

  Characteristic({required this.characteristicName, required this.value});
}
