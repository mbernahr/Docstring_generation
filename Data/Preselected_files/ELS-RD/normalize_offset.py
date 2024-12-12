#  Licensed to the Apache Software Foundation (ASF) under one
#  or more contributor license agreements.  See the NOTICE file
#  distributed with this work for additional information
#  regarding copyright ownership.  The ASF licenses this file
#  to you under the Apache License, Version 2.0 (the
#  "License"); you may not use this file except in compliance
#  with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing,
#  software distributed under the License is distributed on an
#  "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#  KIND, either express or implied.  See the License for the
#  specific language governing permissions and limitations
#  under the License.
from itertools import groupby
from typing import List, Union

from xml_extractions.extract_node_values import Offset


def normalize_offsets(offsets: List[Offset], min_offset_size: int = 0) -> List[Offset]:
    """
    Normalize the provided list of offsets by merging or removing some of them
    Takes care of priority included in the tag label (as `_1`)
    :param offsets: original offsets as list of tuples generated by pattern matching
    :param min_offset_size: minimum offset size, below it is deleted
    :return: cleaned list of tuples
    """

    sorted_offsets: List[Offset] = sorted(offsets, key=lambda o: (o.start, o.end))
    offset_to_keep: List[Offset] = list()
    previous_offset: Union[None, Offset] = None
    # previous_start_offset, previous_end_offset, previous_type_tag = None, None, None

    for current_offset in sorted_offsets:

        # merge 2 tags of the same type which appear as separated but are not really
        if (
            previous_offset is not None
            and previous_offset.end + 1 >= current_offset.start
            and previous_offset.type == current_offset.type
        ):
            previous_offset.end = current_offset.end

        if (previous_offset is not None) and (previous_offset.end < current_offset.end):
            offset_to_keep.append(previous_offset)

        # keep longest tags when they are one on the other
        if (previous_offset is not None) and (previous_offset.end >= current_offset.end):
            # previous_offset.type = tag_priority(previous_offset.type, current_offset.type)
            current_offset = previous_offset

        # delete short offsets (1 - 2 chars)
        if current_offset.end - current_offset.start <= min_offset_size:
            current_offset = None

        previous_offset = current_offset

    if previous_offset is not None:
        offset_to_keep.append(previous_offset)

    return offset_to_keep


def remove_duplicates(data):
    """
    Remove duplicates from the data (normally a list).
    The data must be sortable and have an equality operator
    """
    data = sorted(data)
    return [k for k, v in groupby(data)]
