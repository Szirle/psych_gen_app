import 'package:flutter/material.dart';
import 'package:easy_localization/easy_localization.dart';

class FilterDialog extends StatefulWidget {
  const FilterDialog({super.key});

  @override
  _FilterDialogState createState() => _FilterDialogState();
}

class _FilterDialogState extends State<FilterDialog> {
  String? selectedFilterType = 'filters.type.intensity'.tr();
  double sliderValue = 0.5;
  String selectedOption = 'filters.color.red'.tr();

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      backgroundColor: Colors.white,
      surfaceTintColor: Colors.white,
      title: Text('filters.add_new'.tr()),
      content: SizedBox(
        height: 100,
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
                height: 36,
                width: 160,
                child: DropdownButtonFormField<String>(
                  decoration: InputDecoration(
                    fillColor: Colors.white,
                    focusColor: Colors.white,
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    contentPadding:
                        const EdgeInsets.only(top: 12, left: 12, right: 12),
                  ),
                  value: selectedFilterType,
                  onChanged: (String? newValue) {
                    setState(() {
                      selectedFilterType = newValue;
                    });
                  },
                  items: <String>[
                    'filters.type.intensity'.tr(),
                    'filters.type.color'.tr(),
                  ].map<DropdownMenuItem<String>>((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
                )),
            const SizedBox(width: 24),
            if (selectedFilterType == 'filters.type.intensity'.tr())
              Slider(
                value: sliderValue,
                onChanged: (newValue) {
                  setState(() {
                    sliderValue = newValue;
                  });
                },
                min: 0.0,
                max: 1.0,
              ),
            if (selectedFilterType == 'filters.type.color'.tr())
              SizedBox(
                height: 36,
                width: 160,
                child: DropdownButtonFormField<String>(
                  decoration: InputDecoration(
                    fillColor: Colors.white,
                    focusColor: Colors.white,
                    focusedBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    enabledBorder: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(5.0),
                      borderSide:
                          const BorderSide(color: Colors.black26, width: 1.0),
                    ),
                    contentPadding:
                        const EdgeInsets.only(top: 12, left: 12, right: 12),
                  ),
                  value: selectedOption,
                  onChanged: (String? newValue) {
                    setState(() {
                      selectedOption = newValue!;
                    });
                  },
                  items: <String>[
                    'filters.color.red'.tr(),
                    'filters.color.green'.tr(),
                    'filters.color.blue'.tr(),
                  ].map<DropdownMenuItem<String>>((String value) {
                    return DropdownMenuItem<String>(
                      value: value,
                      child: Text(value),
                    );
                  }).toList(),
                ),
              ),
          ],
        ),
      ),
      actions: <Widget>[
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text('common.cancel'.tr()),
        ),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: Text('common.add'.tr()),
        ),
      ],
    );
  }
}
